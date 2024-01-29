from abc import abstractmethod

import wandb
from torch.utils.data import DataLoader
from dataset import *
from metric import *
from loss import *
from rich.progress import Progress
from utils import *
from model import *
from tqdm import tqdm

from parallel import DataParallelModel, DataParallelCriterion

import pdb
import warnings

warnings.filterwarnings('ignore')

class BasicTrainer:
    def __init__(self, train_data, valid_data, model, logger, opt):
        # save model name for save pth
        self.model_name = model.__class__.__name__
        valid_batch_size = 4
                
        collate = CustomCollate(opt)

        self.train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                        pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
        self.valid_loader = DataLoader(valid_data, batch_size=valid_batch_size, shuffle=False, drop_last=True,
                                        pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
        self.model = model.to(opt.device)

        # optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        # self.DP_class = nn.DataParallel if opt.inference else DataParallelModel

        # others
        self.opt = opt
        self.logger = logger  # can be wandb, logging, or rich.console
        self.progress = None
    
    def data_compress(self, x):
        batch_feat = x['feats']
        batch_label = x['labels']
        noisy_phase = torch.atan2(batch_feat[:, -1, :, :], batch_feat[:, 0, :, :])
        clean_phase = torch.atan2(batch_label[:, -1, :, :], batch_label[:, 0, :, :])
        if self.opt.feat_type == 'normal':
            batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1)
        elif self.opt.feat_type == 'sqrt':
            batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, (
                torch.norm(batch_label, dim=1)) ** 0.5
        elif self.opt.feat_type == 'cubic':
            batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                torch.norm(batch_label, dim=1)) ** 0.3
        elif self.opt.feat_type == 'log_1x':
            batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                      torch.log(torch.norm(batch_label, dim=1) + 1)
        if self.opt.feat_type in ['normal', 'sqrt', 'cubic', 'log_1x']:
            batch_feat = torch.stack((batch_feat * torch.cos(noisy_phase), batch_feat * torch.sin(noisy_phase)),
                                     dim=1)
            batch_label = torch.stack((batch_label * torch.cos(clean_phase), batch_label * torch.sin(clean_phase)),
                                      dim=1)
        return batch_feat, batch_label

    def data_reconstuct(self, x_complex, feat_type='sqrt'):
        x_mag, x_phase = torch.norm(x_complex, dim=1), torch.atan2(x_complex[:, -1, :, :], x_complex[:, 0, :, :])

        if self.opt.feat_type == 'sqrt':
            x_mag = x_mag ** 2
            x_com = torch.stack((x_mag * torch.cos(x_phase), x_mag * torch.sin(x_phase)), dim=1)
        else:
            pass
            # unfinished
        return x_com
    
    def inference_schedule(self, fast_sampling=False):
        """
        Compute fixed parameters in ddpm

        :return:
            alpha:          alpha for training,         size like noise_schedule
            beta:           beta for inference,         size like inference_noise_schedule or noise_schedule
            alpha_cum:      alpha_cum for inference
            sigmas:         sqrt(beta_t^tilde)
            T:              Timesteps
        """
        training_noise_schedule = np.array(self.params.noise_schedule)
        inference_noise_schedule = np.array(
            self.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule  # alpha_t for train
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        sigmas = [0 for i in alpha]
        for n in range(len(alpha) - 1, -1, -1):
            sigmas[n] = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5  # sqrt(beta_t^tilde)
        # print("sigmas", sigmas)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                            talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                    T.append(t + twiddle)  # from schedule to T which as the input of model
                    break
        T = np.array(T, dtype=np.float32)
        return alpha, beta, alpha_cum, sigmas, T    
    
    @abstractmethod
    def run_step(self, x):
        pass

    def train(self):
        prev_cv_loss = float("inf")
        best_cv_loss = float("inf")
        cv_no_impv = 0
        harving = False

        with Progress() as self.progress:
            for epoch in range(self.opt.step, self.opt.n_epoch):
                batch_train = self.progress.add_task(f"[green]training epoch_{epoch}...", total=len(self.train_loader))
                self.model.train()
                for batch in self.train_loader:
                    # cuda
                    for key in batch.keys():
                        try:
                            batch[key] = batch[key].to(self.opt.device)
                        except AttributeError:
                            continue
                    out = self.run_step(batch)
                    self.optim.zero_grad()
                    out['loss'].backward()
                    self.optim.step()
                    self.progress.advance(batch_train, advance=1)
                    if self.opt.wandb:
                        wandb.log({'train_loss': out['loss'].item()})

                mean_valid_loss = self.inference()

                '''Adjust the learning rate and early stop'''
                if self.opt.half_lr > 1:
                    if mean_valid_loss >= best_cv_loss: # change [prev_cv_loss => best_cv_loss]
                        cv_no_impv += 1
                        if cv_no_impv == self.opt.half_lr:
                            harving = True
                        if cv_no_impv >= self.opt.early_stop > 0:
                            self.logger.print("No improvement and apply early stop")
                            return
                    else:
                        cv_no_impv = 0

                if harving == True:
                    optim_state = self.optim.state_dict()
                    for i in range(len(optim_state['param_groups'])):
                        optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                    self.optim.load_state_dict(optim_state)
                    self.logger.print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                    harving = False
                # prev_cv_loss = mean_valid_loss

                if mean_valid_loss < best_cv_loss:
                    self.logger.print(
                        f"best loss is: {best_cv_loss}, current loss is: {mean_valid_loss}, save best_checkpoint.pth")
                    best_cv_loss = mean_valid_loss

                    '''save best checkpoint'''
                    self.save_cpt(epoch,
                                save_path=f'./{self.opt.save_path}/'
                                            f'{self.model_name}_best_{epoch}_{round(best_cv_loss, 4)}.pth')
                self.save_cpt(epoch,
                            save_path=f'./{self.opt.save_path}/'
                                        f'{self.model_name}_{epoch}.pth')
                if self.opt.wandb:
                    wandb.save(f'./{self.opt.save_path}/'
                            f'{self.model_name}_best_{epoch}_{round(best_cv_loss, 4)}.pth')

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        if self.progress:
            loss = self.inference_()
        else:
            with Progress() as self.progress:
                loss = self.inference_()
        return loss
    
    @torch.no_grad()
    def inference_(self):
        loss_list = []
        csig_list, cbak_list, covl_list, pesq_list, ssnr_list, stoi_list = [], [], [], [], [], []
        batch_valid = self.progress.add_task(f"[green]validating...", total=len(self.valid_loader))
        for batch in self.valid_loader:
            # cuda
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.opt.device)
                except AttributeError:
                    continue
            out = self.run_step(batch)  # out['compressed_feats']
            batch_result = compare_complex(out['model_out']['est_comp'], out['compressed_label'],
                                        batch['frame_num_list'],
                                        feat_type=self.opt.feat_type,
                                        is_save_wav=self.opt.save_wav and self.opt.inference,
                                        result_path=f'./{self.opt.save_path}/wav',
                                        wav_name_list=batch['wav_name_list'],
                                        scaling_list=batch['scaling_list'])                                            
   
            loss_list.append(out['loss'].item())
            csig_list.append(batch_result[0])
            cbak_list.append(batch_result[1])
            covl_list.append(batch_result[2])
            pesq_list.append(batch_result[3])
            ssnr_list.append(batch_result[4])
            stoi_list.append(batch_result[5])
            
            self.progress.advance(batch_valid, advance=1)

        self.print_save_logs(loss_list, csig_list, cbak_list, covl_list, pesq_list, ssnr_list, stoi_list)

        return np.mean(loss_list)    
    
    def save_cpt(self, step, save_path):
        """
        save checkpoint, for inference/re-training
        :return:
        """
        torch.save(
            {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict()
            },
            save_path
        )
        
    def print_save_logs(self, *lists):
        loss_list, csig_list, cbak_list, covl_list, pesq_list, ssnr_list, stoi_list = lists
        if self.opt.wandb:
            wandb.log(
                {
                    'test_loss': np.mean(loss_list),
                    'test_mean_csig': np.mean(csig_list),
                    'test_mean_cbak': np.mean(cbak_list),
                    'test_mean_covl': np.mean(covl_list),
                    'test_mean_pesq': np.mean(pesq_list),
                    'test_mean_ssnr': np.mean(ssnr_list),
                    'test_mean_stoi': np.mean(stoi_list)
                }
            )
        else:
            print({
                    'test_loss': np.mean(loss_list),
                    'test_mean_csig': np.mean(csig_list),
                    'test_mean_cbak': np.mean(cbak_list),
                    'test_mean_covl': np.mean(covl_list),
                    'test_mean_pesq': np.mean(pesq_list),
                    'test_mean_ssnr': np.mean(ssnr_list),
                    'test_mean_stoi': np.mean(stoi_list)
            })
            if self.opt.inference: # save result for inference
                with open(f'./{self.opt.save_path}/inference_result.txt', 'w') as fp:
                    fp.write(f'{self.model_name}: {self.opt.resume_from_ckpt}\n\
                            test_mean_stoi: {round(np.mean(stoi_list), 4)}\n\
                            test_mean_pesq: {round(np.mean(pesq_list), 4)}\n\
                            test_mean_csig: {round(np.mean(csig_list), 4)}\n\
                            test_mean_cbak: {round(np.mean(cbak_list), 4)}\n\
                            test_mean_covl: {round(np.mean(covl_list), 4)}\n\
                            test_mean_ssnr: {round(np.mean(ssnr_list), 4)}\n\
                            test_loss:      {round(np.mean(loss_list), 4)}')        
    
    def drow_result(self, temp, condition, spec, batch_label):
        pass
        # f, axs = plt.subplots(2, 4, figsize=(16, 6))
        
        # axs[0, 0].imshow(temp[0, 0, :, :].cpu().numpy())
        # axs[0, 0].set_title('n_real.png')
        # axs[0, 0].axis('off')
        # axs[0, 1].imshow(temp[0, 1, :, :].cpu().numpy())
        # axs[0, 1].set_title('n_imag.png')
        # axs[0, 1].axis('off')
        # axs[0, 2].imshow(condition[0, 0, :, :].cpu().numpy())
        # axs[0, 2].set_title('c_real.png')
        # axs[0, 2].axis('off')
        # axs[0, 3].imshow(condition[0, 1, :, :].cpu().numpy())
        # axs[0, 3].set_title('c_imag.png')
        # axs[0, 3].axis('off')
        
        # axs[1, 0].imshow(spec[0, 0, :, :].cpu().numpy())
        # axs[1, 0].set_title('g_real.png')
        # axs[1, 0].axis('off')
        # axs[1, 1].imshow(spec[0, 1, :, :].cpu().numpy())
        # axs[1, 1].set_title('g_imag.png')
        # axs[1, 1].axis('off')
        # axs[1, 2].imshow(batch_label[0, 0, :, :].cpu().numpy())
        # axs[1, 2].set_title('l_real.png')
        # axs[1, 2].axis('off')
        # axs[1, 3].imshow(batch_label[0, 1, :, :].cpu().numpy())
        # axs[1, 3].set_title('l_imag.png')
        # axs[1, 3].axis('off')
        # plt.savefig(f'asset/data/sample_{self.model_ddpm.__class__.__name__}.jpg', dpi=300, bbox_inches='tight')
        # exit()
            
class VBTrainer(BasicTrainer):
    def __init__(self, train_data, valid_data, model, logger, opt):
        super(VBTrainer, self).__init__(train_data, valid_data, model, logger, opt)
    def run_step(self, x):
        batch_feat, batch_label = self.data_compress(x)
        out = self.model(batch_feat)
        
        loss = com_mag_mse_loss(out['est_comp'], batch_label, x['frame_num_list'])
            
        return {
            'model_out': out,
            'loss': loss,
            'compressed_feats': batch_feat,
            'compressed_label': batch_label
        }

class VBDDPMTrainer(BasicTrainer):
    def __init__(self, train_data, valid_data, model, logger, opt):
        super(VBDDPMTrainer, self).__init__(train_data, valid_data, model, logger, opt)
        
        self.params = opt.params
        beta = np.array(self.params.noise_schedule)  # noise_schedule --> beta
        noise_level = np.cumprod(1 - beta)  # noise_level --> alpha^bar
        self.noise_level = torch.tensor(noise_level.astype(np.float32)).to(self.opt.device)
        
        # load conditon generator
        if opt.c_model: ### check no DP training
            self.c_gen = eval(opt.c_model)() # Base or BaseTNN
            
            checkpoint = torch.load(opt.checkpoints['c_gen'])#"./asset/base_model/Base_best.pth")
            checkpoint['model_state_dict'] = unpack_DP_model(checkpoint['model_state_dict'])
            self.c_gen.load_state_dict(checkpoint['model_state_dict'])
            
            self.c_gen = nn.DataParallel(self.c_gen, device_ids=opt.gpus)
            self.c_gen.to(opt.device)
            self.c_gen.eval()  # newly add 11.20

        if opt.r_model:
            self.refiner = eval(opt.r_model)() # Base or BaseTNN
            
            checkpoint = torch.load(opt.checkpoints['refiner'])#"./asset/selected_model/refiner.pth")
            checkpoint['model_state_dict'] = unpack_DP_model(checkpoint['model_state_dict'])
            self.refiner.load_state_dict(checkpoint['model_state_dict'])
            
            self.refiner = nn.DataParallel(self.refiner, device_ids=opt.gpus)
            self.refiner.to(opt.device)
            self.refiner.eval()
        
    def run_step(self, x):
        batch_feat, batch_label = self.data_compress(x)

        N = batch_label.shape[0]  # Batch size
        t = torch.randint(0, len(self.params.noise_schedule), [N], device=self.opt.device)
        noise_scale = self.noise_level[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # alpha_bar_t       [N, 1, 1, 1]
        noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
        noise = torch.randn_like(batch_label)  # epsilon           [N, 2, T, F]
        noisy_audio = noise_scale_sqrt * (batch_label) + (1.0 - noise_scale) ** 0.5 * noise

        if self.opt.c_model:
            with torch.no_grad():
                condition = self.c_gen(batch_feat)['est_comp']
        else:
            condition = batch_feat

        out = self.model(noisy_audio, condition, t)
        
        loss = com_mse_loss(out['est_noise'], noise, x['frame_num_list'])
            
        return {
            'model_out': out,
            'loss': loss,
            'compressed_feats': batch_feat,
            'compressed_label': batch_label
        }
        
    # def inference_ddpm(self):
    @torch.no_grad()
    def inference_(self):
        loss_list, csig_list, cbak_list, covl_list, pesq_list, ssnr_list, stoi_list = [], [], [], [], [], [], []
        alpha, beta, alpha_cum, sigmas, T = self.inference_schedule(fast_sampling=self.params.fast_sampling)
        
        batch_valid_ddpm = self.progress.add_task(f"[green]validating...", total=len(self.valid_loader))
        for batch in self.valid_loader:
            # cuda
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.opt.device)
                except AttributeError:
                    continue
                
            ''' For training, compute test loss'''
            if self.opt.inference == False:
                out = self.run_step(batch)
                loss_list.append(out['loss'].item())
                
            # discard run_step function ==> restore
            batch_feat, batch_label = self.data_compress(batch)
            if self.opt.c_model:
                condition = self.c_gen(batch_feat)['est_comp']
            else:
                condition = batch_feat
            # spec = torch.randn_like(condition)

            # '''dr_diffuse_new: change spec from gaussian to y_T (start)'''
            # noise_scale = self.noise_level[-1]
            # noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
            # noise = torch.randn_like(batch_feat)  # epsilon           [N, 2, T, F]
            # # y_T = noise_scale_sqrt * (batch_feat) + (1.0 - noise_scale) ** 0.5 * noise
            # # spec = y_T
            # c_T = noise_scale_sqrt * (condition) + (1.0 - noise_scale) ** 0.5 * noise
            # spec = c_T
            # '''dr_diffuse_new: change spec from gaussian to y_T (end)'''

            '''dr_diffuse_new: change spec from gaussian to c_T (start, consider fast-sampling)'''
            noise_scale = alpha_cum[-1]
            noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
            noise = torch.randn_like(batch_feat)  # epsilon           [N, 2, T, F]
            c_T = noise_scale_sqrt * (condition) + (1.0 - noise_scale) ** 0.5 * noise
            spec = c_T
            '''dr_diffuse_new: change spec from gaussian to c_T (end, consider fast-sampling)'''

            temp = spec  # for draw
            N = batch_label.shape[0]  # batch size
            for n in tqdm(range(len(alpha) - 1, -1, -1), leave=True):

                t = torch.tensor([T[n]], device=spec.device).repeat(N)
                out = self.model(spec, condition, t)

                c1 = 1 / alpha[n] ** 0.5  # for ddpm sampling
                c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5  # for ddpm sampling
                spec = c1 * (spec - c2 * out['est_noise'])
                if n > 0:  # + random noise
                    noise = torch.randn_like(spec)
                    spec += sigmas[n] * noise

                    # add condition guidance
                    if self.opt.c_guidance:
                        noise_scale = torch.Tensor([alpha_cum[n]]).unsqueeze(1).unsqueeze(2).unsqueeze(
                            3).cuda()  # alpha_bar_t [N, 1, 1, 1]
                        noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
                        noise = torch.randn_like(condition).cuda()  # epsilon           [N, 2, T, F]
                        noisy_condition = noise_scale_sqrt * condition + (1.0 - noise_scale) ** 0.5 * noise  # c_t
                        spec = 0.5 * spec + 0.5 * noisy_condition
            if self.opt.r_model:
                spec = 0.5 * spec + 0.5 * batch_feat
                spec = self.refiner(spec)['est_comp']

            '''run code in below can draw the difference between initial gaussian, condition (noisy), generated, and ground truth'''
            if self.opt.drow_result:
                self.drow_result(temp, condition, spec, batch_label)
            
            
            batch_result = compare_complex(spec, batch_label,
                                        batch['frame_num_list'],
                                        feat_type=self.opt.feat_type,
                                        is_save_wav=self.opt.save_wav and self.opt.inference,
                                        result_path=f'./{self.opt.save_path}/wav',
                                        wav_name_list=batch['wav_name_list'],
                                        scaling_list=batch['scaling_list'])   

            csig_list.append(batch_result[0])
            cbak_list.append(batch_result[1])
            covl_list.append(batch_result[2])
            pesq_list.append(batch_result[3])
            ssnr_list.append(batch_result[4])
            stoi_list.append(batch_result[5])

            self.progress.advance(batch_valid_ddpm, advance=1)
        
        ''' Save logs to wandb or terminal, and save result to txt file'''
        self.print_save_logs(loss_list, csig_list, cbak_list, covl_list, pesq_list, ssnr_list, stoi_list)
            
        return np.mean(loss_list)
            
class RefinerTrainer(BasicTrainer):
    def __init__(self, train_data, valid_data, model, console, opt):
        super(RefinerTrainer, self).__init__(train_data, valid_data, model, console, opt)

        # c_gen
        self.c_gen = eval(opt.c_model)() #Base()
        
        checkpoint = torch.load(opt.checkpoints['c_gen'])#"./asset/selected_model/c_gen.pth")
        checkpoint['model_state_dict'] = unpack_DP_model(checkpoint['model_state_dict'])
        self.c_gen.load_state_dict(checkpoint['model_state_dict'])
        
        self.c_gen = nn.DataParallel(self.c_gen, device_ids=opt.gpus)        
        self.c_gen.to(opt.device)
        self.c_gen.eval()

        # ddpm
        self.ddpm_model = eval(opt.d_model)(opt.params) #DiffuSE(opt.params)
        
        checkpoint = torch.load(opt.checkpoints['ddpm_model'])#f"./asset/selected_model/ddpm.pth")
        checkpoint['model_state_dict'] = unpack_DP_model(checkpoint['model_state_dict'])
        self.ddpm_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.ddpm_model = nn.DataParallel(self.ddpm_model, device_ids=opt.gpus)        
        self.ddpm_model.to(opt.device)
        self.ddpm_model.eval()
        
        # for inference
        self.params = opt.params
        beta = np.array(self.params.noise_schedule)  # noise_schedule --> beta
        noise_level = np.cumprod(1 - beta)  # noise_level --> alpha^bar
        self.noise_level = torch.tensor(noise_level.astype(np.float32)).to(self.opt.device)
        self.alpha, self.beta, self.alpha_cum, self.sigmas, self.T = self.inference_schedule(
            fast_sampling=self.params.fast_sampling)

    def run_step(self, x):
        """
        (1) load stft version from data_loader;
        (2) compress;
        (3) generate c via c_gen;
        (4) feed c to ddpm and generate augmented data;
        (5) feed generated data to refiner
        (6) train refiner
        :param x:
        :return:
        """
        with torch.no_grad():
            batch_feat, batch_label = self.data_compress(x)  # (2)
            N = batch_label.shape[0]  # batch size

            c = self.c_gen(batch_feat)['est_comp']  # (3)

            # (4)
            noise_scale = self.alpha_cum[-1]
            noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
            noise = torch.randn_like(batch_feat)  # epsilon           [N, 2, T, F]
            x_T = noise_scale_sqrt * (c) + (1.0 - noise_scale) ** 0.5 * noise
            spec = x_T

            for n in range(len(self.alpha) - 1, -1, -1):

                t = torch.tensor([self.T[n]], device=spec.device).repeat(N)
                out = self.ddpm_model(spec, c, t)

                c1 = 1 / self.alpha[n] ** 0.5  # for ddpm sampling
                c2 = self.beta[n] / (1 - self.alpha_cum[n]) ** 0.5  # for ddpm sampling
                spec = c1 * (spec - c2 * out['est_noise'])
                if n > 0:  # + random noise
                    noise = torch.randn_like(spec)
                    spec += self.sigmas[n] * noise

                    # add condition guidance
                    noise_scale = torch.Tensor([self.alpha_cum[n]]).unsqueeze(1).unsqueeze(2).unsqueeze(
                        3).cuda()  # alpha_bar_t [N, 1, 1, 1]
                    noise_scale_sqrt = noise_scale ** 0.5  # sqrt(alpha_bar_t) [N, 1, 1, 1]
                    noise = torch.randn_like(c).cuda()  # epsilon           [N, 2, T, F]
                    c_t = noise_scale_sqrt * c + (1.0 - noise_scale) ** 0.5 * noise  # c_t
                    spec = 0.5 * spec + 0.5 * c_t
        spec = 0.5 * spec + 0.5 * batch_feat
        out = self.model(spec) # model = refiner
        loss = com_mag_mse_loss(out['est_comp'], batch_label, x['frame_num_list'])
        return {
            'model_out': out,
            'loss': loss,
            'compressed_feats': batch_feat,
            'compressed_label': batch_label
        }            