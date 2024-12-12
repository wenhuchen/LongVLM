import torch
import os
import subprocess
import datetime


def init_dist(args):
    num_gpus = torch.cuda.device_count()
    args.rank = int(os.getenv('SLURM_PROCID', '0'))
    args.local_rank = args.rank % (num_gpus // args.num_gpus_per_rank)
    args.world_size = int(os.getenv('SLURM_NTASKS', '1'))
    args.local_world_size = num_gpus // args.num_gpus_per_rank

    os.environ['RANK'] = str(args.rank)
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(args.local_world_size)

    if 'MASTER_ADDR' not in os.environ:
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ['MASTER_ADDR'] = addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '22110'

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=args.rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(days=2)
    )
    torch.cuda.set_device(args.local_rank)