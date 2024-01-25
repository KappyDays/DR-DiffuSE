import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import wandb
import torch
import random
import argparse
from rich.console import Console
from model import *
from trainer import *
from utils import *

import pdb

def main(rank, opt):
    # make save_path
    if opt.save_path[-1] == '/':
        opt.save_path = opt.save_path[:-1]
    os.makedirs(opt.save_path, exist_ok=True)
    
    # Set multi GPU
    opt.local_gpu_id = 'cpu'
    if not opt.cpu:
        init_distributed_training(rank, opt)
        opt.local_gpu_id = opt.gpu
    
    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    '''logger'''
    # logger = get_logger(f'./asset/log/{opt.model}.log')
    # logger.info(opt)

    '''load data'''
    tr_data = VBDataset(
        f'./data/{opt.data_dir}/noisy_trainset_wav',
        f'./data/{opt.data_dir}/clean_trainset_wav',
        'train',
        opt)
    cv_data = VBDataset(
        f'./data/{opt.data_dir}/noisy_testset_wav',
        f'./data/{opt.data_dir}/clean_testset_wav',
        'valid',
        opt)


    console = Console(color_system='256', style=None)
    if opt.wandb:
        wandb.init(project="dr_diffuse_BaseTNN", group="DDP_train")
        # wandb.init(project="dr_diffuse", settings=wandb.Settings(start_method="fork"))
    else:
        if is_main_process():
            print(console.print("wandb forbidden!"))
    
    if is_main_process():
        console.print(f'total {tr_data.__len__()} train data, total {cv_data.__len__()} eval data.')

    
    '''load model'''
    model = eval(opt.model)()

    ## load model for inference
    
    '''load trainer'''
    trainer = VBTrainer(tr_data, cv_data, model, console, opt)

    if opt.inference:
        device = 'cpu' if opt.cpu else f'cuda:{opt.local_gpu_id}'
        checkpoint = torch.load(f"./results/Base/BaseTNN/BaseTNN_best_0_1.4697.pth", map_location=device)
        # checkpoint = torch.load(f"./results/Base/BaseTNN_Residual/BaseTNN_best_37_0.0763.pth", map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.to(device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model.to(opt.device)
        trainer.inference()
    else:
        trainer.train()
    
    if opt.wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2023, help='manual seed')

    parser.add_argument('--model', type=str, default="Base", help='Base')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-7, help="weight decay")
    parser.add_argument('--half_lr', type=int, default=3, help='decay learning rate to half scale')
    parser.add_argument('--early_stop', type=int, default=5, help='early stop training')

    parser.add_argument('--win_size', type=int, default=320)
    parser.add_argument('--fft_num', type=int, default=320)
    parser.add_argument('--win_shift', type=int, default=160)
    parser.add_argument('--chunk_length', type=int, default=48000)
    parser.add_argument('--feat_type', type=str, default='sqrt', help='normal/sqrt/cubic/log_1x')

    parser.add_argument('--wandb', action='store_true', help='load wandb or not')
    parser.add_argument('--wandb_project_name', type=str)
    
    
    parser.add_argument('--data_dir', type=str, default="voicebank", help='choose data directory')
    parser.add_argument('--save_path', type=str, default="results/Base/default_train", help='save path')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--wandb', type=str, default="dr_diffuse_Base", help='typing wandb project name')
    # parser.add_argument('--test_args', type=str, default="234s", help='typing wandb project name')
    parser = add_args_parser(parser)

    args = parser.parse_args()
    
    
    args.ngpus_per_node = torch.cuda.device_count()
    args.gpu_ids = list(range(args.ngpus_per_node))
    # args.device = torch.device('cuda')

    print(f'workspace:{os.getcwd()} available training device:{args.gpu_ids}')
        
    # main(args)
    # torch.multiprocessing.set_start_method("spawn")
    if args.cpu:
        main(None, args)
    else:
        torch.multiprocessing.spawn(main,
                    args=(args,),
                    nprocs=args.ngpus_per_node,
                    join=True)