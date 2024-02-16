from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
import os
import torch.nn as nn
import random
import torch
import numpy as np
import librosa
import torchaudio
import pdb
import time, datetime
import warnings

warnings.filterwarnings('ignore')

class ToTensor(object):
    def __call__(self, x, tensor_type='float'):
        if tensor_type == 'float':
            return torch.FloatTensor(x)
        elif tensor_type == 'int':
            return torch.IntTensor(x)


class BatchInfo(object):
    def __init__(self, noisy, clean, frame_num_list, wav_len_list):
        self.feats = noisy
        self.labels = clean
        self.frame_num_list = frame_num_list
        self.wav_len_list = wav_len_list


class CustomCollate(object):
    def __init__(self):
        self.win_size = 320
        self.fft_num = 320
        self.win_shift = 160
        # self.device = opt.device

    @staticmethod
    def normalize(x):
        return x / np.max(abs(x))

    # return {
    #     'noisy_speech': noisy,
    #     'clean_speech': clean,
    #     'frame_num': frame_num,
    #     'wav_len': wav_len
    # }

    def collate_fn(self, batch):
        noisy_list, clean_list, frame_num_list, wav_len_list, wav_name_list, scaling_list = [], [], [], [], [], []
        to_tensor = ToTensor()
        for sample in batch:
            c = torch.sqrt(len(sample['noisy_speech']) / torch.sum(sample['noisy_speech'] ** 2.0))
            scaling_list.append(c)
            noisy_list.append(to_tensor(sample['noisy_speech'] * c))
            clean_list.append(to_tensor(sample['clean_speech'] * c))
            frame_num_list.append(sample['frame_num'])
            wav_len_list.append(sample['wav_len'])
            wav_name_list.append(sample['wav_name'])
        noisy_list = nn.utils.rnn.pad_sequence(noisy_list, batch_first=True)
        clean_list = nn.utils.rnn.pad_sequence(clean_list, batch_first=True)  # [b, chunk_length]
        noisy_list = torch.stft(
            noisy_list,
            n_fft=self.fft_num,
            hop_length=self.win_shift,
            win_length=self.win_size,
            window=torch.hann_window(self.fft_num),
            return_complex=False
        ).permute(0, 3, 2, 1)  # [b, 2, T, F] real tensor, return_complex = false
        clean_list = torch.stft(
            clean_list,
            n_fft=self.fft_num,
            hop_length=self.win_shift,
            win_length=self.win_size,
            window=torch.hann_window(self.fft_num),
            return_complex=False
        ).permute(0, 3, 2, 1)  # [b, 2, T, F]

        return {
            'feats': noisy_list,
            'labels': clean_list,
            'frame_num_list': frame_num_list,
            'wav_len_list': wav_len_list,
            'wav_name_list': wav_name_list,
            'scaling_list': scaling_list
        }


class VBDataset(Dataset):
    def __init__(self, noisy_root, clean_root, data_type):
        super(VBDataset, self).__init__()
        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.chunk_length = 48000
        self.win_size = 320
        self.fft_num = 320
        self.win_shift = 160
        self.raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]
        print('이전:',len(self.raw_paths))
        self.raw_paths = self.raw_paths[:len(self.raw_paths)//2]
        print('이후:',len(self.raw_paths))
        

        assert data_type in ['train', 'valid', 'test']
        self.data_type = data_type  # determine train or test
        
        ori_sr = 16000 if False else 48000
        self.resample = torchaudio.transforms.Resample(ori_sr, 16000)

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):
        wav_name = self.raw_paths[index]
        # pdb.set_trace()
        noisy, _ = torchaudio.load(os.path.join(self.noisy_root, self.raw_paths[index]))
        noisy = self.resample(noisy).squeeze()
        # noisy2 = torchaudio.transforms.Resample(noisy, n_sr, 16000)
        clean, _ = torchaudio.load(os.path.join(self.clean_root, self.raw_paths[index]))
        clean = self.resample(clean).squeeze()
        # clean2 = torchaudio.transforms.Resample(clean, c_sr, 16000)
        # noisy2, _ = librosa.load(os.path.join(self.noisy_root, self.raw_paths[index]), sr=16000)
        # clean2, _ = librosa.load(os.path.join(self.clean_root, self.raw_paths[index]), sr=16000)
        if self.data_type == 'train':
            if len(noisy) > self.chunk_length:
                wav_start = random.randint(0, len(noisy) - self.chunk_length)
                noisy = noisy[wav_start:wav_start + self.chunk_length]
                clean = clean[wav_start:wav_start + self.chunk_length]
        wav_len = len(noisy)
        frame_num = (len(noisy) - self.win_size + self.fft_num) // self.win_shift + 1
        # return noisy, clean, frame_num, wav_len
        return {
            'noisy_speech': noisy,
            'clean_speech': clean,
            # 'noisy_speech2': noisy2,
            # 'clean_speech2': clean2,
            'frame_num': frame_num,
            'wav_len': wav_len,
            'wav_name': wav_name
        }
tr_data = VBDataset(
    f'./data/voicebank/noisy_trainset_remove_val_wav',
    f'./data/voicebank/clean_trainset_remove_val_wav',
    'train')

# tr_data = VBDataset(
#     f'./data/chime4/noisy_trainset_remove_val_wav',
#     f'./data/chime4/clean_trainset_remove_val_wav',
#     'train')

train_loader = DataLoader(tr_data, batch_size=64, shuffle=True, drop_last=True,
                                pin_memory=True, collate_fn=CustomCollate().collate_fn, num_workers=0)#, prefetch_factor=2)        



st = time.time()
for datas in tqdm(train_loader):
    pass
ed = time.time()

print(f'{datetime.timedelta(seconds=ed-st)}')
