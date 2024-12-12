import os
import json
import time
import argparse
import torch

from PIL import Image
from tqdm import tqdm
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer, AutoConfig
from internvl.train.dataset import build_transform, dynamic_preprocess
from eval.mm_niah.tools import get_input, init_dist

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

FILEPATH = 'eval/mm_niah/mm_niah.json'
DATA_ROOT = 'dataset/benchmark/MM-NIAH'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def load_image(image_file, dynamic_image_size=True, input_size=448, max_num=12, return_additional_info=False):
    if not return_additional_info:
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(is_train=False, input_size=input_size)
        if dynamic_image_size:
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    else:
        image = Image.open(image_file).convert('RGB')
        orig_size = image.size

        transform = build_transform(is_train=False, input_size=input_size)
        if dynamic_image_size:
            images, boxes = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num, return_box=True)
        else:
            images = [image]
            boxes = [(0,0,orig_size[0],orig_size[1]), ]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values, images, boxes, orig_size


def build_model(args):

    num_gpus = torch.cuda.device_count()

    visible_devices = [i for i in range(args.local_rank, num_gpus, args.local_world_size)]

    if len(visible_devices) > 1:
        device_map = {}
        config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)


        num_gpus_for_vit = 1
        num_gpus_for_llm = len(visible_devices) - num_gpus_for_vit


        num_layers = config.llm_config.num_hidden_layers

        num_layers_per_gpu = num_layers // num_gpus_for_llm + 1
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu + num_gpus_for_vit, len(visible_devices) - 1)
            device_map[f'language_model.model.layers.{i}'] = visible_devices[device_idx]

        num_layers = config.vision_config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus_for_vit + 1
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu, num_gpus_for_vit - 1)
            device_map[f'vision_model.encoder.layers.{i}'] = visible_devices[device_idx]

        device_map['vision_model.embeddings'] = visible_devices[0]
        device_map['local_posid.weight'] = visible_devices[0]
        device_map['mlp1'] = visible_devices[num_gpus_for_vit - 1]
        # InternLM2
        device_map['language_model.model.tok_embeddings'] = visible_devices[num_gpus_for_vit]
        device_map['language_model.model.norm'] = visible_devices[-1]
        device_map['language_model.output'] = visible_devices[-1]
        # Qwen2
        device_map['language_model.model.embed_tokens'] = visible_devices[num_gpus_for_vit]
        device_map['language_model.model.norm'] = visible_devices[-1]
        device_map['language_model.lm_head'] = visible_devices[-1]

    else:
        device_map = {'': visible_devices[0]}

    if args.rank == 0:
        for k, v in device_map.items():
            print(k, v)

    model = InternVLChatModel.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    # tokenizer.model_max_length = 32000
    tokenizer.model_max_length = 256000

    return model, tokenizer



def main(args):
    init_dist(args)

    task = args.task
    model_name = os.path.basename(args.checkpoint)
    model, tokenizer = build_model(args)

    print(
        f"Rank [{args.rank}] "
        f"Begin to eval model {args.checkpoint} on task {task}, "
        f"devices: {set([p.device for p in model.parameters()])}"
    )

    temp_dir = f"temp_{model_name}_{task}"
    ans_file_name = f'{model_name}_{task}.jsonl'
    ans_file_path = os.path.join(args.outputs_dir, temp_dir, f"{args.rank}_{args.world_size}_{ans_file_name}")

    if args.rank == 0:
        os.makedirs(os.path.join(args.outputs_dir, temp_dir), exist_ok=True)
    torch.distributed.barrier()

    with open(args.question_file, 'r') as file:
        lines = file.readlines()

    skip_idx = set()
    if os.path.exists(ans_file_path):
        with open(ans_file_path) as file:
            ans_lines = file.readlines()

        for ans_line in ans_lines:
            skip_idx.add(json.loads(ans_line)['question_id'])

    ans_file = open(ans_file_path, 'a')
    lines = lines[args.rank::args.world_size]
    lines = [json.loads(line) for line in lines]
    lines = sorted(lines, key=lambda x:x['meta']['context_length'])

    lines = lines[::-1]

    if args.rope_pos_id_stride is not None:
        rope_pos_id_stride = args.rope_pos_id_stride
        print(f'USE {rope_pos_id_stride=} from config')
    else:
        model_config = model.config
        rope_pos_id_stride = getattr(model_config, 'rope_pos_id_stride', None)
        print(f'USE {rope_pos_id_stride=}')

    oom_cnt = 0
    print(f'Rank {args.rank} {len(skip_idx)=}')
    for sample in tqdm(lines, desc=f"Processing {ans_file_name}", disable=args.rank!=0):
        if sample['id'] in skip_idx:
            continue

        if oom_cnt >= 20:
            print(f"[Rank {args.rank}] early stops because of successive failures. {oom_cnt=}")
            ans_file.write(json.dumps({
                "question_id": sample['id'],
                "question": question,
                "answer": sample['answer'],
                "response": 'None',
                'context_length':sample['meta']['context_length'],
                'placed_depth':sample['meta']['placed_depth']
            }) + "\n")
            ans_file.flush()
            continue

        context, images_list, question, answer = get_input(sample)
        images_list = [os.path.join(args.image_folder, i) for i in images_list]

        all_boxes, num_tiles, for_posid_image_list, orig_sizes = [], [], [], []

        qs = f'{context}{question}'

        if len(images_list) > 0:
            pixel_values = []
            num_patches_list = []
            for img in images_list:
                ## TODO: check the max_num
                curr_pixel_values, images, boxes, orig_size = load_image(img, dynamic_image_size=True, max_num=5, return_additional_info=True)

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
                max_new_tokens=320 if 'counting' in task else 16,
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
                    rope_pos_id_stride=rope_pos_id_stride)
            oom_cnt = 0
        except torch.cuda.OutOfMemoryError as e:
            print(f"[Rank {args.rank}] OutOfMemoryError occurs! totoal_tokens={sample['meta']['context_length']}, error: {e}")
            outputs = 'None'
            oom_cnt += 1
            try:
                torch.cuda.empty_cache()
            except:
                exit(0)

        outputs = outputs.strip()
        print(f"[{current_time()}] [Rank {args.rank}] totoal_tokens={sample['meta']['context_length']}, {outputs=}, answer={sample['answer']}")

        ans_file.write(json.dumps({
            "question_id": sample['id'],
            "question": question,
            "answer": sample['answer'],
            "response": outputs,
            'context_length': sample['meta']['context_length'],
            'placed_depth': sample['meta']['placed_depth'],
        }) + "\n")
        ans_file.flush()
        skip_idx.add(sample['id'])

    print(f"[{current_time()}] Rank {args.rank} Finish")
    ans_file.close()

    torch.distributed.barrier()

    if args.rank == 0:
        print(f'cat {args.outputs_dir}/{temp_dir}/* >  {args.outputs_dir}/{model_name}_{task}.jsonl')
        os.system(f'cat {args.outputs_dir}/{temp_dir}/* >  {args.outputs_dir}/{model_name}_{task}.jsonl')
        print(f'python eval/mm_niah/calculate_scores.py --outputs-dir {args.outputs_dir}')
        os.system(f'python eval/mm_niah/calculate_scores.py --outputs-dir {args.outputs_dir}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for MM-NIAH")
    parser.add_argument('--checkpoint', type=str, default='OpenGVLab/InternVL-Chat-V1-5')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--outputs-dir', type=str, default='')
    parser.add_argument('--num-gpus-per-rank', type=int, default=2)
    parser.add_argument('--image-folder', type=str, default='')
    parser.add_argument('--question-file', type=str, default='')
    parser.add_argument('--rope_pos_id_version', type=str, default='default')
    parser.add_argument('--rope_pos_id_stride', type=int, default=None)
    args = parser.parse_args()
    print(f'{args.rope_pos_id_version=}')

    args.outputs_dir = os.path.join(args.outputs_dir, args.task)

    print(f"{args=}")

    with open(FILEPATH) as file:
        meta = json.load(file)

    if not args.image_folder:
        args.image_folder = os.path.join(DATA_ROOT, meta[args.task]['root'])
    if not args.question_file:
        args.question_file = os.path.join(DATA_ROOT, meta[args.task]['annotation'])
    print("Start evaluation on task", args.task)
    main(args)
