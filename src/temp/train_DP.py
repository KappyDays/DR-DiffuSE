import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import wandb
import torch
import torch.nn as nn
import random
import argparse
from rich.console import Console
from model import *
from trainer_ori import *
from utils import *

import pdb

def main(opt):
    # make save_path
    opt.save_path = opt.save_path[:-1] if opt.save_path[-1] == '/' else opt.save_path
    os.makedirs(opt.save_path, exist_ok=True)
    if opt.save_wav:
        os.makedirs(opt.save_path + '/wav', exist_ok=True)
    
    
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
        wandb.init(project=opt.wpn if opt.wpn else "dr_diffuse_BaseTNN_default",
                   name=opt.wrn if opt.wrn else "Base_residual_resume",
                   resume=True if opt.resume_from_ckpt else False)
        # wandb.init(project="dr_diffuse", settings=wandb.Settings(start_method="fork"))
    else:
        print(console.print("wandb forbidden!"))
    
    console.print(f'total {tr_data.__len__()} train data, total {cv_data.__len__()} eval data.')

    '''load model'''
    model = eval(opt.model)()

    '''load trainer'''
    trainer = VBTrainer(tr_data, cv_data, model, console, opt)

    '''Use DataParallel'''
    if torch.cuda.is_available() and (args.gpus != None) > 1:
        trainer.model = nn.DataParallel(trainer.model, device_ids=args.gpus)
        trainer.model = trainer.model.to(opt.device)   
        
    '''load checkpoint for inference or resume training'''
    opt.step = 0
    if opt.resume_from_ckpt is not None:
        load_file = wandb.restore(opt.resume_from_ckpt) if opt.wandb_resume else opt.resume_from_ckpt
        checkpoint = torch.load(load_file)
        
        ''' DataParallel model인 경우 module.을 제거'''
        if checkpoint['model_state_dict'].keys()[0][:7] == 'module.':
            unpacked_state_dict = unpack_DP_model(checkpoint['model_state_dict'])
            trainer.model.load_state_dict(unpacked_state_dict)
        else:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        opt.step = checkpoint['step']
    
    '''Check inference or training'''
    if opt.inference:
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
    parser.add_argument('--batch_size', type=int, default=4)
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
    
    parser = add_custom_args(parser)

    args = parser.parse_args()
    
    if torch.cuda.is_available() and (args.gpus != None) > 0:
        # remove any device which doesn't exists
        args.gpus = [int(d) for d in args.gpus if 0 <= int(d) < torch.cuda.device_count()]
        # set args.gpus[0] (the master node) as the current device
        torch.cuda.set_device(args.gpus[0])
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
 
    # args.device = torch.device('cuda:0')
    print(f'workspace:{os.getcwd()} training device:{args.device} num_gpus:{args.gpus}')
        
    main(args)