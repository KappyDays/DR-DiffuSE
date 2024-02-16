import torch
import torch.nn as nn
from .source.aia_nets import *
from functools import partial

class DB_AIAT(nn.Module): #dual_aia_trans_merge_crm
    def __init__(self, noise_schedual_length):
        super(DB_AIAT, self).__init__()
        
        self.preprocess = Preprocess()
        self.time_embedding = TimeEmbedding(noise_schedual_length)
        
        self.en_ri = dense_encoder(in_ch=2, out_ch=1)
        self.en_mag = dense_encoder(in_ch=1, out_ch=1) # for magnitude
        
        self.aia_trans_merge = AIA_Transformer_merge(128, 64, num_layers=4)
        self.aham = AHAM_ori(input_channel=64)
        self.aham_mag = AHAM_ori(input_channel=64)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder(masking=True)

    def forward(self, x, c, t):
        x = self.preprocess(x, c)
        # t = self.time_embedding(t)
        
        batch_size, _, seq_len, _ = x.shape
        x_r_input, x_i_input = x[:,0,:,:], x[:,1,:,:]
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :])
        x_mag = x_mag_ori.unsqueeze(dim = 1)
        # ri/mag components enconde+ aia_transformer_merge
        x_ri = self.en_ri(x) #BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag, x_last_ri, x_outputlist_ri  = self.aia_trans_merge(x_mag_en, x_ri)  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri) #BCT
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim=1)
        # magnitude and ri components interaction

        x_mag_out = x_mag_mask * x_mag_ori
        x_r_out, x_i_out = (x_mag_out * torch.cos(x_phase_ori) + x_real), (x_mag_out * torch.sin(x_phase_ori)+ x_imag)

        x_com_out = torch.stack((x_r_out, x_i_out), dim=1)

        return x_com_out

class TimeEmbedding(nn.Module):  # from diffwave
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 128)
        self.projection2 = nn.Linear(128, 128)
        self.silu = nn.SiLU()
        # def silu(x):
        #     return x * torch.sigmoid(x)
    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = self.silu(x)
        x = self.projection2(x)
        x = self.silu(x)
        return x

    def _lerp_embedding(self, t):
        # print(t.shape)
        # print(self.embedding.shape) # [50, 128]
        low_idx = torch.floor(t).long()  # [8]
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]  # [8, 128]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx).unsqueeze(1)  # [8, 128]

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table
        
class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()
        self.conv = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x, x_init):
        return self.conv(torch.cat((x, x_init), dim=1))
    
class dense_encoder(nn.Module):
    def __init__(self, in_ch, out_ch, width=64):
        super(dense_encoder, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)
        
        self.tp1 = nn.Linear(128, self.in_channels)
        self.tp2 = nn.Linear(128, 64)

    def forward(self, x, t=None):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        # out = self.inp_prelu(self.inp_norm(self.inp_conv(x + self.tp1(t).unsqueeze(-1).unsqueeze(-1))))  # [b, 64, T, F]
        out = self.enc_dense1(out)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        # x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out + self.tp2(out).unsqueeze(-1).unsqueeze(-1))))  # [b, 64, T, F]
        return x

class dense_decoder(nn.Module):
    def __init__(self, width=64, masking=False):
        super(dense_decoder, self).__init__()
        self.masking = masking
        
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width =width
        self.dec_dense1 = DenseBlock(80, 4, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)
        
        if self.masking == True:
            self.mask1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (1,1)),
                nn.Sigmoid()
            )
            self.mask2 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size= (1,1)),
                nn.Tanh()
            )
            self.maskconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,1))
            #self.maskrelu = nn.ReLU(inplace=True)
            self.maskrelu = nn.Sigmoid()
            
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out)))))

        out = self.out_conv(out)
        out.squeeze(dim=1)
        
        if self.masking == True: # mask 
            out = self.mask1(out) * self.mask2(out)
            out = self.maskrelu(self.maskconv(out))
        return out
    
    
class SPConvTranspose2d(nn.Module): #sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module): #dilated dense block
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out    