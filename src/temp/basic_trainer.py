from abc import abstractmethod

from dataset import *
from torch.utils.data import DataLoader
from utils import *

class BasicTrainer:
    def __init__(self, train_data, valid_data, model, logger, opt):
        # save model name for save pth
        self.model_name = model.__class__.__name__

        collate = CustomCollate(opt)

        # data with DDP
        if opt.cpu:
            self.train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                           pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
            self.valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False, drop_last=True,
                                           pin_memory=True, collate_fn=collate.collate_fn, num_workers=opt.num_workers)
            self.model = model
        else:
            self.train_loader = self.ddp_data_loader(train_data, opt, 'train', collate.collate_fn)
            self.valid_loader = self.ddp_data_loader(valid_data, opt, 'valid', collate.collate_fn)
            self.model = model.cuda(opt.local_gpu_id) #to(opt.device)
            self.model = DistributedDataParallel(module=model, device_ids=[opt.local_gpu_id], output_device=opt.local_gpu_id)        
            
        # train_sampler = DistributedSampler(train_data, shuffle=True)
        # batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opt.batch_size, drop_last=True)
        # self.train_loader = DataLoader(train_data, batch_sampler=batch_sampler_train, 
        #                                pin_memory=True, collate_fn=collate.collate_fn,
        #                                num_workers=opt.num_workers)
        
        # valid_sampler = DistributedSampler(valid_data, shuffle=False)
        # batch_sampler_valid = torch.utils.data.BatchSampler(valid_sampler, opt.batch_size, drop_last=True)
        # self.valid_loader = DataLoader(valid_data, batch_sampler=batch_sampler_valid,
        #                                pin_memory=True, collate_fn=collate.collate_fn,
        #                                num_workers=opt.num_workers)

        # optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

        # others
        self.opt = opt
        self.logger = logger  # can be wandb, logging, or rich.console
        self.progress = None

    @abstractmethod
    def run_step(self, x):
        pass

    @abstractmethod
    def train(self):
        pass

    def save_cpt(self, step, save_path):
        """
        save checkpoint, for inference/re-training
        :return:
        """
        torch.save(
            {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.model.state_dict()
            },
            save_path
        )
        
    def ddp_data_loader(self, dataset, opt, data_type, collate_fn=None):
        shuffle = True if data_type == 'train' else False
        data_sampler = DistributedSampler(dataset, shuffle=shuffle)
        batch_sampler = BatchSampler(data_sampler, opt.batch_size, drop_last=True)
        data_loader = DataLoader(dataset, batch_sampler=batch_sampler, 
                                        pin_memory=True, collate_fn=collate_fn,
                                        num_workers=opt.num_workers)
        return data_loader

    def ddp_inference(self, output_list):
        output = np.mean(output_list)
        
        # 분산 환경에서는 결과를 모으고 평균을 계산
        output = torch.distributed.all_gather(torch.Tensor(output))

        # 모든 결과를 평균
        output = torch.stack(output).mean(dim=0)
        
        return output