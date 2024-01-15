import argparse

import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--port', type=int, default=2033)
    parser.add_argument('--root', type=str, default='./cifar')
    parser.add_argument('--local_rank', type=int)
    return parser

def init_distributed_training(rank, opts):
    # 1. setting for distributed training
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    torch.distributed.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:' + str(opts.port),
                            world_size=opts.ngpus_per_node,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()

    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print('opts :',opts)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print