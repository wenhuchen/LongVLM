import os
import orjson as json
import time
import argparse
import torch
from tqdm import tqdm
from eval.mm_niah.tools import init_dist
from eval.mm_niah.eval_mm_niah import load_image, build_model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def write_jsonl(l, path):
    with open(path, 'ab') as f:
        for d in l:
            f.write(json.dumps(d) + b'\n')


def read_jsonl(path):
    lines = []
    with open(path, 'rb') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def main(args):
    init_dist(args)
    if args.rank == 0 and os.path.exists(args.savepath) and args.overwrite:
       os.remove(args.savepath)
    
    skip_idx = set()
    if os.path.exists(args.savepath) and not args.overwrite:
        ans_lines = read_jsonl(args.savepath)
        for line in ans_lines:
            skip_idx.add(line['id'])
    
    annotation = read_jsonl(args.file)
    if args.val_length != -1:
        annotation = annotation[:args.val_length]
    annotation = annotation[args.rank::args.world_size]
    model, tokenizer = build_model(args)
        
    if args.rope_pos_id_stride is not None:
        rope_pos_id_stride = args.rope_pos_id_stride
    else:
        model_config = model.config
        rope_pos_id_stride = getattr(model_config, 'rope_pos_id_stride', None)
    
    if args.rank == 0:
        print(
            f"{args=}"
            f"Rank [{args.rank}] "
            f"devices: {set([p.device for p in model.parameters()])}"
            f'USE {rope_pos_id_stride=}'
        )

    torch.distributed.barrier()
        
    for sample in tqdm(annotation, desc=f"Processing", disable=args.rank!=0):
        id = sample['id']
        if id in skip_idx:
            continue
        qs = sample['conversations'][0]['value']
        answer = sample['conversations'][1]['value']
        images_list = sample['image']
        context_length = sample['metadata']['context_length'] if 'metadata' in sample and 'context_length' in sample['metadata'] else -1

        all_boxes, num_tiles, for_posid_image_list, orig_sizes = [], [], [], []
        
        if len(images_list) > 0:
            pixel_values = []
            num_patches_list = []
            for img in images_list:
                curr_pixel_values, images, boxes, orig_size = load_image(img, dynamic_image_size=True, max_num=12, return_additional_info=True)
                
                for_posid_image_list.append(images)
                all_boxes.append(boxes)
                num_tiles.append(len(images))
                orig_sizes.append(orig_size)
                
                pixel_values.append(curr_pixel_values)
                num_patches_list.append(len(curr_pixel_values))
            pixel_values = torch.cat(pixel_values)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        else:
            pixel_values = None
            num_patches_list = []

        try:
            generation_config=dict(
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_token,
            )
            if args.rope_pos_id_version == 'default':
                outputs = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    question=qs,
                    generation_config=generation_config)
            else:
                outputs = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    question=qs,
                    generation_config=generation_config,
                    num_tiles=[num_tiles, ],
                    all_boxes=[all_boxes, ],
                    orig_sizes=[orig_sizes, ],
                    image_list=[for_posid_image_list, ],
                    rope_pos_id_version=args.rope_pos_id_version,
                    rope_pos_id_stride=rope_pos_id_stride
                )
            oom_cnt = 0
            
            write_jsonl([{
                'id': id, 
                'an': answer,
                'pr': outputs,
                'le': context_length
            }], args.savepath)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"[Rank {args.rank}] OutOfMemoryError occurs! total_tokens={context_length}, error: {e}")
            outputs = 'None'
            oom_cnt += 1
            torch.cuda.empty_cache()
        
        print(f"[{current_time()}] [Rank {args.rank}] total_tokens={context_length}, {outputs=}")

    print(f"[{current_time()}] Rank {args.rank} Finish")

    torch.distributed.barrier()
    
    os.system(f'python eval/auto/calculate_scores.py --outputs-dir {args.outputs_dir} --result-file result_{args.task} --score-file score_{args.task} --task {args.task}')

task2token = {
    'chartqa': 32,
    'clevr': 8,
    'deepform': 32, 
    'docvqa': 32,
    'dvqa': 8,
    'gqa': 8,
    'infovqa': 32,
    'kleistercharity': 32,
    'ocrvqa': 128,
    'okvqa': 16,
    'svqa': 8,
    'tabfact': 8,
    'textcaps': 64,
    'textvqa': 64,
    'visualmrc': 256,
    'vizwiz': 32,
    'wikitablequestions': 128,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--num-gpus-per-rank', type=int, default=2)
    parser.add_argument('--task', type=str)
    parser.add_argument('--file', type=str)
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--outputs-dir', type=str)
    parser.add_argument('--val-length', type=int, default=-1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--rope_pos_id_version', type=str, default='default')
    parser.add_argument('--rope_pos_id_stride', type=int, default=None)
    args = parser.parse_args()

    args.savepath = os.path.join(args.outputs_dir, f"result_{args.task}.jsonl")
    if args.root and not args.root.endswith('/') and not args.root.endswith(':'):
        args.root += '/'
    args.max_token = task2token[args.task]
    os.makedirs(args.outputs_dir, exist_ok=True)
    
    main(args)
