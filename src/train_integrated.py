import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import wandb
import torch
import torch.nn as nn
import random
import argparse
from rich.console import Console
from model import *
# from trainer_ori import *
from utils import *
from trainer_integrated import *
from parallel import DataParallelModel, DataParallelCriterion
# https://jjdeeplearning.tistory.com/32
import pdb

''' changeable params '''
checkpoints = {'c_gen': './results/Base/BaseTNN/Base_residual/BaseTNN_best_29_0.0733.pth', 
               'refiner': './results/Base/BaseTNN/Base_residual/BaseTNN_best_29_0.0733.pth', 
               'ddpm_model': None}
Base_model_list = ['Base', 'BaseTNN']
DDPM_model_list = ['DiffuSE', 'DiffuSEC', 'Refiner']

wandb_project_name = 'dr_diffuse_ddpm'
wandb_run_name = 'BaseTNN'
''' changeable params '''

def main(opt):
    '''make save_path'''
    opt.save_path = opt.save_path[:-1] if opt.save_path[-1] == '/' else opt.save_path
    os.makedirs(opt.save_path, exist_ok=True)
    if opt.save_wav:
        os.makedirs(opt.save_path + '/wav', exist_ok=True)
    
    '''fix seed'''
    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    '''logger'''
    # logger = get_logger(f'./asset/log/{opt.model}.log')
    # logger.info(opt)

    '''wandb'''
    console = Console(color_system='256', style=None)
    if opt.wandb:
        wandb.init(project=opt.wpn if opt.wpn else wandb_project_name,
                   name=opt.wrn if opt.wrn else wandb_run_name,
                   resume=True if opt.resume_from_ckpt else False)
        # wandb.init(project="dr_diffuse", settings=wandb.Settings(start_method="fork"))
    else:
        print(console.print("wandb forbidden!"))
        
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
    
    ''' for mismatched training'''
    # cv_data = VBDataset( 
    #     './data/chime4/noisy',
    #     './data/chime4/clean',
    #     'valid',
    #     opt)    
    
    console.print(f'total {tr_data.__len__()} train data, total {cv_data.__len__()} eval data.')

    '''load model and trainer'''
    opt.checkpoints = checkpoints
    if opt.training_step == 'base':
        model = eval(opt.model)()
        trainer = VBTrainer(tr_data, cv_data, model, console, opt)
    elif opt.training_step == 'ddpm':
        opt.params = AttrDict(
            ours=False,
            fast_sampling=opt.fast_sampling,
            noise_schedule=np.linspace(1e-4, 0.05, 200).tolist(),
            inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.35],
        )
        model = eval(opt.model)(opt.params)
        trainer = VBDDPMTrainer(tr_data, cv_data, model, console, opt)
    elif opt.training_step == 'refiner':
        model = eval(opt.model)()
        trainer = RefinerTrainer(tr_data, cv_data, model, console, opt)

    '''Use DataParallel'''
    if torch.cuda.is_available() and (args.gpus != None):
        if len(args.gpus) >= 2: # gpu 개수가 2개 이상인 경우만 DP 사용
            trainer.model = nn.DataParallel(trainer.model, device_ids=args.gpus)
            trainer.model = trainer.model.to(opt.device)   
        
    '''load checkpoint for inference or resume training'''
    opt.step = 0
    if opt.resume_from_ckpt is not None:
        load_file = wandb.restore(opt.resume_from_ckpt) if opt.wandb_resume else opt.resume_from_ckpt
        checkpoint = torch.load(load_file)
        
        ''' 학습을 CPU에서 진행하고, DataParallel model인 경우 module.을 제거'''
        if opt.device == torch.device('cpu'):
            if checkpoint['model_state_dict'].keys()[0][:7] == 'module.':
                unpacked_state_dict = unpack_DP_model(checkpoint['model_state_dict'])
                trainer.model.load_state_dict(unpacked_state_dict)
            else:
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        opt.step = checkpoint['step'] + 1
    
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
    
    parser.add_argument('--model', type=str, default="Base",
                        help='--Choose Training Model--\n\
                        Training Step 1(Base): [Base or BaseTNN]\n\
                        Training Step 2(DDPM): [DiffuSE or DiffuSEC]\n\
                        Training Step 3(Refiner): [Base or BaseTNN]')
    parser.add_argument('--training_step', type=str, required=True, choices=["base", "ddpm", "refiner"], help='choose training step')
    
    '''for Base traning, select None
        for DDPM training, select (c_model and r_model)
        for refiner training, select (c_model and d_model)'''
    parser.add_argument('--c_model', type=str, choices=["Base", "BaseTNN"], help='choose condition generate(c_gen) model')
    parser.add_argument('--r_model', type=str, choices=["Base", "BaseTNN"], help='choose refine model')
    parser.add_argument('--d_model', type=str, default="DiffuSE", choices=["DiffuSE", "DiffuSEC"], help='choose diffusion model')

    '''for ddpm or refiner training or inference'''
    parser.add_argument('--c_guidance', action='store_true', help='choose to use explicit condition guidance during inference')
    parser.add_argument('--fast_sampling', action='store_true', help='for ddpm or refiner')
    
    '''for refiner training'''
    parser.add_argument('--from_base', action='store_true', help='for refiner')
    
    # parser.add_argument('--c_gen', action='store_true', help='choose to use condition generated from condition generator')
    # parser.add_argument('--refine', action='store_true', help='choose to use condition generated from condition generator')
    
    ''' hyper params'''    
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epoch') # for refiner, if start from base, choose a small value such as 2
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

    ''' utils '''
    parser.add_argument('--wandb', action='store_true', help='load wandb or not')
    
    parser.add_argument('--data_dir', type=str, default="voicebank", help='choose data directory')
    parser.add_argument('--save_path', type=str, default="results/default_save", help='save path')
    parser.add_argument('--save_wav', action='store_true', help='save wav result')
    parser.add_argument('--inference', action='store_true', help='set inference mode')
    parser.add_argument('--gpus', type=int, nargs="+", help='set gpus')
    parser.add_argument('--resume_from_ckpt', type=str, help='resume training')
    parser.add_argument('--wandb_resume', action='store_true', help='resume training with wandb')
    
    parser.add_argument('--wpn', type=str, help='wandb project name')
    parser.add_argument('--wrn', type=str, help='wandb run name')
    # parser = add_custom_args(parser)

    args = parser.parse_args()
    
    '''set device'''
    if torch.cuda.is_available() and (args.gpus != None):
        # remove any device which doesn't exists
        args.gpus = [int(d) for d in args.gpus if 0 <= int(d) < torch.cuda.device_count()]
        # set args.gpus[0] (the master node) as the current device
        torch.cuda.set_device(args.gpus[0])
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
 
    print(f'workspace:{os.getcwd()} training device:{args.device} num_gpus:{args.gpus}')
        
    main(args)