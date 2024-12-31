import os
import io
import json
import time
import argparse
import torch

from PIL import Image
from tqdm import tqdm
from internvl.model.internvl_chat import InternVLChatModel
from internvl.model.internlm2.modeling_internlm2 import InternLM2LinearScalingRotaryEmbedding,InternLM2DynamicNTKScalingRotaryEmbedding
from transformers import AutoTokenizer, AutoConfig
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.train.dataset import preprocess_internlm
from internvl.model.internvl_chat.modeling_internvl_chat import get_rope_pos_id
from internvl.patch import replace_internlm2_attention_class
from eval.mm_niah.tools import get_input, init_dist
from copy import deepcopy
import torch.distributed as dist

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

FILEPATH = 'eval/mm_niah/mm_niah_long.json'
DATA_ROOT = 'dataset/benchmark/MM-NIAH'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

try:
    from petrel_client.client import Client
    client = Client("~/petreloss_zy.conf")
except:
    print('petrel_client is not installed. Using PIL to load images.')

def get_img(img_url):
    img = io.BytesIO(client.get(img_url))
    return img


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


ROOT2="dataset/benchmark/MM-NIAH/mm_niah_test/images"
def load_image(image_file, dynamic_image_size=True, input_size=448, max_num=12, return_additional_info=False):

    if 's3:' in image_file:
        image_file = get_img(image_file)
    else:
        image_file=os.path.join(ROOT2,image_file)
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
    print(f"{device_map=}")

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
    tokenizer.model_max_length = 1400000

    return model, tokenizer


def main(args):
    init_dist(args)

    task = args.task
    model_name = os.path.basename(args.checkpoint)
    if args.ring_attn==True:
        replace_internlm2_attention_class('ring')
    model, tokenizer = build_model(args)
    if args.interp=='linear':
        assert args.rope_pos_id_stride==256
        for layer in model.language_model.model.layers:
            layer.attention.rotary_emb=InternLM2LinearScalingRotaryEmbedding(
                    layer.attention.head_dim,
                    max_position_embeddings=32000,
                    base=layer.attention.config.rope_theta,
                    scaling_factor=args.factor,
                )
        print(f"replaced to {args.factor} linear")
    elif args.interp=='ntk':
        assert args.rope_pos_id_stride==256
        for layer in model.language_model.model.layers:
            layer.attention.rotary_emb=InternLM2DynamicNTKScalingRotaryEmbedding(
                    layer.attention.head_dim,
                    max_position_embeddings=32000,
                    base=layer.attention.config.rope_theta,
                    scaling_factor=args.factor,
                )
    print(
        f"Rank [{args.rank}] "
        f"Begin to eval model {args.checkpoint} on task {task}, "
        f"devices: {set([p.device for p in model.parameters()])}"
    )

    temp_dir = f"temp_{model_name}_{task}"
    ans_file_name = f'{model_name}_{task}_ring_attn.jsonl'
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
    if args.ring_attn ==True:
        pass
    else:
        lines = lines[args.rank::args.world_size]
    lines = [json.loads(line) for line in lines]
    lines = sorted(lines, key=lambda x:x['meta']['context_length'])
    if args.ring_attn==True:
        model.attn_type='ring'
    if args.rope_pos_id_stride is not None:
        rope_pos_id_stride = args.rope_pos_id_stride
    else:
        model_config = model.config
        rope_pos_id_stride = getattr(model_config, 'rope_pos_id_stride', None)
    print(f'USE {rope_pos_id_stride=}')

    oom_cnt = 0
    print(f'Rank {args.rank} {len(skip_idx)=}')
    for sample in tqdm(lines, desc=f"Processing {ans_file_name}", disable=args.rank!=0):
        if sample['id'] in skip_idx:
            continue
        sample = [sample]
        dist.broadcast_object_list(sample, src=0)
        sample = sample[0]

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
        try:
            context, images_list, question, answer = get_input(sample)
        except Exception as e:
            print(e)
            print(f"{sample['id']=}")
            continue

        new_images_list = []
        for i in images_list:
            if 's3:' in i:
                new_images_list.append('langchao2:' + i)
            else:
                new_images_list.append(os.path.join(args.image_folder, i))
        images_list = new_images_list

        all_boxes, num_tiles, for_posid_image_list, orig_sizes = [], [], [], []

        qs = f'{context}{question}'

        assert '<image>' in qs
        
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
                max_new_tokens=320 if 'counting' in task else 16,
                # max_new_tokens=120
            )
            
            def convert(context, images_list, question, answer ):
                qs = f'{context}{question}'
                conversation={'id':sample['id'],'image':images_list,'conversations':[{'from':'human','value':qs},{'from':'gpt','value':str(answer)}]}
                return conversation

            conv=convert(context, images_list, question, answer )
            question_conv=deepcopy(conv)
            question_conv['conversations'][1]['value']=''
            answer_conv=deepcopy(conv)
            answer_conv['conversations'][0]['value']=''
            template='internlm2-chat'
            position_ids=[]
            q=conv['conversations'][0]['value']
            cnt=q.count('<image>')
            num_image_tokens = [256 * num_tile for num_tile in num_tiles]
            num_image=len(images_list)
            ret=preprocess_internlm(template, [deepcopy(conv['conversations'])],tokenizer,num_image_tokens, group_by_length=False,use_packed_ds=True, ds_name='niah',num_image=num_image)
            ret_q=preprocess_internlm(template, [deepcopy(question_conv['conversations'])],tokenizer,num_image_tokens, group_by_length=False,use_packed_ds=True, ds_name='niah',num_image=num_image)
            input_ids=ret['input_ids'].cuda()
            input_ids_q=ret_q['input_ids'].cuda()
            prompt_length=input_ids_q.shape[1]
            labels_length=input_ids.shape[1]-input_ids_q.shape[1]
            answer_ids=input_ids[:,prompt_length:]
            labels=ret['labels'].cuda()

            IMG_START_TOKEN='<img>'
            IMG_END_TOKEN='</img>'
            # 
            for i in range(input_ids.shape[0]):
                position_ids.append(torch.tensor(get_rope_pos_id(ret, num_tiles=[num_tiles, ][i], dtype=torch.float32,
                                           rope_pos_id_version=args.rope_pos_id_version,
                                           position_id=torch.arange(0,input_ids.shape[1]),
                                           boxes=[all_boxes,][i],
                                           orig_size=None,
                                           images=[for_posid_image_list, ][i],
                                           IMG_START_TOKEN=IMG_START_TOKEN,
                                           IMG_END_TOKEN=IMG_END_TOKEN, rope_pos_id_stride=rope_pos_id_stride)).cuda())

                dist.barrier()
            position_ids=torch.stack(position_ids).to(input_ids.device)
            if input_ids.shape[1]%(2*dist.get_world_size())!=0:
                num_padding = 2*dist.get_world_size()-input_ids.shape[1]%(2*dist.get_world_size())

                padding_shape = (input_ids.shape[0], num_padding)
                input_padding = torch.full(padding_shape, 1, dtype=input_ids.dtype, device=input_ids.device)
                label_padding = torch.full(padding_shape, -100, dtype=labels.dtype, device=labels.device)

                input_ids = torch.cat([input_ids, input_padding], dim=1)
                labels = torch.cat([labels, label_padding], dim=1)

                max_pos_id = position_ids.max() + 1
                pos_padding = torch.arange(max_pos_id, max_pos_id + num_padding, device=input_ids.device)
                pos_padding = pos_padding.unsqueeze(0).expand(input_ids.shape[0], -1)
                position_ids = torch.cat([position_ids, pos_padding], dim=1)
            attention_mask=torch.tensor([[0,input_ids.shape[1]]],dtype=torch.int32).cuda()
            img_flag=torch.ones(pixel_values.shape[0]).to(pixel_values.device)
            IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
            img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            model.img_context_token_id = img_context_token_id
            with torch.no_grad():
                logits=model.forward(pixel_values=pixel_values,input_ids=input_ids,attention_mask=attention_mask,labels=labels,position_ids=position_ids.cuda(),image_flags=img_flag).logits
                
            pred = logits.argmax(dim=-1)
            def undo_extract_local(gathered_value, world_size, dim=1):
                value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
                reordered_chunks = [None] * (2 * world_size)
                for i in range(world_size):
                    reordered_chunks[i] = value_chunks[i * 2]
                    reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
                return torch.cat(reordered_chunks, dim=dim)
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            gathered_logits = [torch.zeros_like(pred) for _ in range(world_size)]

            dist.all_gather(gathered_logits, pred)

            gathered_logits = torch.stack(gathered_logits).unsqueeze(0)
            gathered_logits=gathered_logits.view(1,-1)
            pred = undo_extract_local(gathered_logits, world_size)
            origin_pred=pred.clone()
            pred = pred[:, prompt_length - 1 : prompt_length + labels_length - 1]
            
            outputs=tokenizer.decode(pred[0][:-1])
            origin_outputs=tokenizer.decode(origin_pred[0])
            if isinstance(answer,int):
                correct=chr(ord('A') + answer)==outputs.strip()
            else:
                correct = (pred[:,:-1] == answer_ids[:,:-1].to(pred.device)).all().item()
            dist.barrier()
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
        if dist.get_rank()==0:
            print(f"[{current_time()}] [Rank {args.rank}] totoal_tokens={sample['meta']['context_length']}, {outputs=}, answer={sample['answer']},correct= {correct}")
            ans_file.write(json.dumps({
                "question_id": sample['id'],
                "question": question,
                "answer": sample['answer'],
                "response": outputs,
                'context_length': sample['meta']['context_length'],
                'placed_depth': sample['meta']['placed_depth'],
                'correct': correct
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
    parser.add_argument('--interp',type=str,default='None')
    parser.add_argument('--factor',type=float,default=1)
    parser.add_argument('--ring_attn', action='store_true')
    args = parser.parse_args()
    print(f'{args.rope_pos_id_version=}')
    print(f'{args.ring_attn=}')
    args.outputs_dir = os.path.join(args.outputs_dir, args.task)
    
    print(f"{args=}")

    with open(FILEPATH) as file:
        meta = json.load(file)


    if not 'long' in args.task:

        args.question_file = os.path.join(DATA_ROOT, meta[args.task]['annotation'])
    else:

        args.image_folder = meta[args.task]['root']
        args.question_file = meta[args.task]['annotation']
    print("Start evaluation on task", args.task)
    print(f'{args.image_folder=}, {args.question_file=}')
    main(args)
