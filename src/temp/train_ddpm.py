import os
import wandb
import torch
import random
import argparse
from rich.console import Console
from model import *
from ddpm_trainer import *
from utils import *

# os.environ['WANDB_MODE'] = 'offline'

def main(opt):
    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    if not os.path.exists('./asset/log/'):
        os.mkdir('./asset/log/')

    '''logger'''
    logger = get_logger(f'./asset/log/{opt.model}.log')
    logger.info(opt)

    console = Console(color_system='256', style=None)
    if opt.wandb:
        wandb.init(project="dr_diffuse")
        # wandb.init(project="dr_diffuse_new", settings=wandb.Settings(start_method="fork"))
    else:
        console.print("wandb forbidden!")

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

    console.print(f'total {tr_data.__len__()} train data, total {cv_data.__len__()} eval data.')

    '''load model'''
    opt.params = AttrDict(
        ours=False,
        fast_sampling=opt.fast_sampling,
        noise_schedule=np.linspace(1e-4, 0.05, 200).tolist(),
        inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35],
    )
    model = eval(opt.model)(opt.params)

    '''load trainer'''
    trainer = VBDDPMTrainer(tr_data, cv_data, model, console, logger, opt)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2023, help='manual seed')

    parser.add_argument('--model', type=str, default="DiffuSE", help='Base/DiffuSEC/DiffuSE/...')
    parser.add_argument('--c_gen', action='store_true', help='choose to use condition generated from condition generator')
    parser.add_argument('--refine', action='store_true', help='choose to use condition generated from condition generator')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=8)
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

    parser.add_argument('--fast_sampling', action='store_true', help='')
    parser.add_argument('--data_dir', type=str)

    args = parser.parse_args()
    args.device = torch.device('cuda:0')

    print(f'workspace:{os.getcwd()}, training device:{args.device}')

    main(args)
