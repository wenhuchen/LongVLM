import itertools
from argparse import ArgumentParser

import time
import json, os
from torch.utils.data import Dataset, DataLoader
import torch

from eval.milebench.utils import MileBenchDataset
from internvl.train.dataset import build_transform, dynamic_preprocess
from eval.mm_niah.tools import init_dist
from eval.mm_niah.eval_mm_niah import build_model
from tqdm import tqdm
import io
from PIL import Image


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='dataset/benchmark/MileBench')
    parser.add_argument('--dataset_name', default=None, type=str)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--bsz', default=1, type=int)
    parser.add_argument('--combine_image', default=None, type=int, help='Use combined N images for evaluation.')
    parser.add_argument('--model_configs', default='configs/model_configs.yaml')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num-gpus-per-rank', type=int, default=2)
    parser.add_argument('--max_context_len', type=int, default=512000)
    parser.add_argument('--n_tokens_per_image', type=int, default=256)
    parser.add_argument('--dynamic-image-size', action='store_true')
    parser.add_argument('--max-dynamic-patch', type=int, default=12)
    parser.add_argument('--resize-image', action='store_true')
    parser.add_argument('--rope_pos_id_version', type=str, default='default')
    parser.add_argument('--rope_pos_id_stride', type=int, default=None)

    args = parser.parse_args()
    print(f'{args.rope_pos_id_version=}')

    args.output_dir = os.path.join(args.output_dir, f"{args.dataset_name}")
    args.output_pth = os.path.join(args.output_dir, f"pred.json")

    return args


SIZE_MAP = {
    (320, 480): (420, 480),
    (266, 480): (420, 480),
    (480, 318): (480, 420),
    (480, 392): (480, 420),
    (360, 480): (420, 480),
    (480, 360): (480, 420),
    (392, 480): (420, 480),
    (480, 276): (480, 272),
    (480, 320): (480, 420),
    (480, 352): (480, 420),
    (480, 268): (480, 420),
    (1920, 1080): (1152, 648),
    (1280, 720): (1152, 648),
    (1920, 896): (1280, 600)
}


def load_image(image_file, dynamic_image_size=True, input_size=448, max_num=12, return_additional_info=False):
    if not return_additional_info:
        image = Image.open(image_file).convert('RGB')
        if args.resize_image and image.size in SIZE_MAP:
            image = image.resize(SIZE_MAP[image.size])
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

        if args.resize_image and image.size in SIZE_MAP:
            image = image.resize(SIZE_MAP[image.size])
        transform = build_transform(is_train=False, input_size=input_size)
        if dynamic_image_size:
            images, boxes = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num, return_box=True)
        else:
            images = [image]
            boxes = [(0, 0, orig_size[0], orig_size[1]), ]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values, images, boxes, orig_size


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def split_data(data):
    '''
    Split the data by the images number
    ex: {
        2: [sample1, ...]
        3: [sample2, ...]
    }
    '''
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict

def save(results, accelerator, args):
    if accelerator.is_main_process:
        if os.path.exists(args.output_pth):
            if not args.overwrite:
                print(f'{args.output_pth} exists. Please pass `overwrite=True` to avoid unwanted overwriting.')
                exit(0)
        json.dump(results, open(args.output_pth, 'w'), ensure_ascii=False, indent=4)

def main(args):
    init_dist(args)

    task = args.dataset_name
    model_name = os.path.basename(args.checkpoint)
    model, tokenizer = build_model(args)

    print(f"Rank [{args.rank}] "
          f"Begin to eval model {args.checkpoint} on task {args.dataset_name}, "
          f"devices: {set([p.device for p in model.parameters()])}")
    
    if args.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)


    ######################### Loading Data #########################
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    combine_image = args.combine_image
    dataset_dir = os.path.join(data_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, 'images')

    core_annotation = json.load(
        open(os.path.join(
            dataset_dir, f'{dataset_name}_combined_{combine_image}.json'
            if combine_image and combine_image!=1 else f'{dataset_name}.json'
            )
        )
    )
    # split data by images number
    data_dict = split_data(core_annotation['data'])
    # # sort by the number of images descending
    # data_dict = dict(sorted(data_dict.items(), key=lambda x: x[0], reverse=True))
    # sort by the number of images ascending
    data_dict = dict(sorted(data_dict.items(), key=lambda x: x[0], reverse=False))
    
    # if args.dataset_name in ['CharacterOrder', 'StateChange']:
    #     print(f"Number of image {max(data_dict.keys())} items are removed")
    #     data_dict.pop(max(data_dict.keys()))
    ################################################################


    ###################### Start Generating ########################

    print('Initialization Finished')
    print(f'Predicting {dataset_name} Using {model_name}')
    generation_config = dict(
        do_sample=False,
        num_beams=1,
        max_new_tokens=32,
    )
    outputs = []
    for n_img, sub_data in data_dict.items():
        print(f'Proceeding {n_img}-length images samples | Num: {len(sub_data)}')
        lc_dataset = MileBenchDataset(
            annotation=sub_data,
            task_instructions=core_annotation['meta_data']['task_instruction'],
            img_dir=img_dir,
            max_context_len=args.max_context_len,
            n_tokens_per_image=args.n_tokens_per_image,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            combine_image=combine_image,
        )
        lc_dataloader = DataLoader(
            dataset=lc_dataset,
            sampler=InferenceSampler(len(lc_dataset)),
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            collate_fn=lc_dataset.collate_fn
        )

        if args.rope_pos_id_stride is not None:
            rope_pos_id_stride = args.rope_pos_id_stride
        else:
            model_config = model.config
            rope_pos_id_stride = getattr(model_config, 'rope_pos_id_stride', None)
        print(f'USE {rope_pos_id_stride=}')

        # start inference
        
        for batch in tqdm(lc_dataloader) if args.rank else lc_dataloader:
            for _, (sample_id, question, images_list, gt_response) in enumerate(zip(batch['id'], batch['question'], batch['image_path'], batch['gt_response'])):

                all_boxes, num_tiles, for_posid_image_list, orig_sizes = [], [], [], []

                if len(images_list) > 0:
                    pixel_values = []
                    num_patches_list = []
                    for img in images_list:
                        curr_pixel_values, images, boxes, orig_size = load_image(
                            img, 
                            dynamic_image_size=args.dynamic_image_size,
                            max_num=args.max_dynamic_patch, return_additional_info=True)

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
                    if args.rope_pos_id_version == 'default':
                        pred = model.chat(
                            tokenizer=tokenizer,
                            pixel_values=pixel_values,
                            question=question,
                            num_patches_list=num_patches_list,
                            generation_config=generation_config
                        )  # list[dict], with the key "answer" added to each item
                    else:
                        pred = model.chat(
                            tokenizer=tokenizer,
                            pixel_values=pixel_values,
                            num_patches_list=num_patches_list,
                            question=question,
                            generation_config=generation_config,
                            num_tiles=[num_tiles, ],
                            all_boxes=[all_boxes, ],
                            orig_sizes=[orig_sizes, ],
                            image_list=[for_posid_image_list, ],
                            rope_pos_id_version=args.rope_pos_id_version,
                            rope_pos_id_stride=rope_pos_id_stride
                        )
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[Rank {args.rank}] OutOfMemoryError occurs! error: {e}")
                    pred = 'None'
                    torch.cuda.empty_cache()

                pred = pred.strip()
                print(f"[{current_time()}] [Rank {args.rank}], {pred=}, answer={gt_response}")
                pred = {
                    'sample_id': sample_id,
                    'question': question,
                    'gt_response': gt_response,
                    'gen_kwargs': dict(generation_config),
                    'pred_response': pred.strip(),
                }
                outputs.append(pred)
       
    # gather all results
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [
        _ for _ in itertools.chain.from_iterable(merged_outputs)
    ]

    if args.rank == 0:
        json.dump(merged_outputs, open(args.output_pth, 'w'), ensure_ascii=False, indent=4)

        print(f'evaluating {task} ...')
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{task}_{time_prefix}.json'
        results_file = os.path.join(args.output_dir, results_file)
        json.dump(merged_outputs, open(results_file, 'w'))
        print('Results saved to {}'.format(results_file))

        cmd_string = f'python eval/milebench/evaluate.py  \
                    --data-dir {args.data_dir} \
                    --dataset {args.dataset_name} \
                    --result-dir {args.output_dir} '
        print(cmd_string)
        os.system(cmd_string)
    
    torch.distributed.barrier()

if __name__ == '__main__':
    args = parse_args()
    main(args)
