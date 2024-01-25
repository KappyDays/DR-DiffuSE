import logging
import soundfile as sf
from metric import compareone
import numpy as np
import argparse

import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import BatchSampler
from torch.nn.parallel import DistributedDataParallel

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # sh.setLevel(level_dict[verbosity])
    # logger.addHandler(sh)

    return logger


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self
    

def add_custom_args(parser):
    parser.add_argument('--data_dir', type=str, default="voicebank", help='choose data directory')
    parser.add_argument('--save_path', type=str, default="results/default_save", help='save path')
    parser.add_argument('--save_wav', action='store_true', help='save wav result')
    parser.add_argument('--inference', action='store_true', help='set inference mode')
    parser.add_argument('--gpus', type=int, nargs="+", help='set gpus')
    parser.add_argument('--resume_from_ckpt', type=str, help='resume training')
    parser.add_argument('--wandb_resume', action='store_true', help='resume training with wandb')
    
    parser.add_argument('--wpn', type=str, help='wandb project name')
    parser.add_argument('--wrn', type=str, help='wandb run name')
    return parser

def unpack_DP_model(state_dict):
    ''' unpack DataParallel model state_dict to single GPU model state_dict'''
    unpacked_state_dict = {}
    for key in state_dict:
        if key[:7] == 'module.':
            unpacked_state_dict[key[7:]] = state_dict[key]#.to(opt.device)
        else:
            unpacked_state_dict[key] = state_dict[key]#.to(opt.device)
    return unpacked_state_dict
            
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
    # setup_for_distributed(opts.rank == 0)
    # print('opts :',opts)


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
    
def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()

def is_main_process():
    return get_rank() == 0

# def save_on_master(*args, **kwargs):
#     if is_main_process():
#         torch.save(*args, **kwargs)