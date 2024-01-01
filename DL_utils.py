"""
@author: Yuqiang (Ethan) Heng
"""

import numpy as np
import math
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from beam_utils import UPA_DFT_codebook, unravel_index
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import time
import pickle

class Joint_Tx_Rx_Analog_Beamformer_DFT(Module):
    def __init__(self, num_antenna_Tx: int, num_antenna_Rx: int, num_beam_Tx: int, num_beam_Rx: int, noise_power: float, norm_factor: float=1.0) -> None:
        super(Joint_Tx_Rx_Analog_Beamformer_DFT, self).__init__()
        self.num_antenna_Tx = num_antenna_Tx
        self.num_antenna_Rx = num_antenna_Rx
        self.num_beam_Tx = num_beam_Tx
        self.num_beam_Rx = num_beam_Rx
        self.register_buffer('noise_power',torch.tensor([noise_power]).float())
        self.register_buffer('norm_factor',torch.tensor([norm_factor]).float())
        n_ant_per_dim_Rx = int(np.sqrt(num_antenna_Rx))
        n_beam_per_dim_Rx = int(np.sqrt(num_beam_Rx))
        n_ant_per_dim_Tx = int(np.sqrt(num_antenna_Tx))
        n_beam_per_dim_Tx = int(np.sqrt(num_beam_Tx))
        self.register_buffer('Rx_codebook',torch.from_numpy(UPA_DFT_codebook(n_azimuth=n_beam_per_dim_Rx,n_elevation=n_beam_per_dim_Rx,n_antenna_azimuth=n_ant_per_dim_Rx,n_antenna_elevation=n_ant_per_dim_Rx,spacing=0.5).T).to(torch.cfloat))
        self.register_buffer('Tx_codebook',torch.from_numpy(UPA_DFT_codebook(n_azimuth=n_beam_per_dim_Tx,n_elevation=n_beam_per_dim_Tx,n_antenna_azimuth=n_ant_per_dim_Tx,n_antenna_elevation=n_ant_per_dim_Tx,spacing=0.5).T).to(torch.cfloat))        
        
    def forward(self, h: Tensor) -> Tensor:
        # h is n_batch x num_antenna_Rx x num_antenna_Tx
        Tx_codebook = self.Tx_codebook # num_antenna_Tx x num_beam_Tx
        Rx_codebook = self.Rx_codebook # num_antenna_Rx x num_beam_Rx
        y = torch.matmul(h,Tx_codebook) # n_batch x num_antenna_Tx x num_beam_Tx       
        noise_real = torch.normal(0,1, size=y.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise_imag = torch.normal(0,1, size=y.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise = torch.complex(noise_real, noise_imag)
        y_s = torch.matmul(Rx_codebook.conj().transpose(0,1),y)
        y_n = torch.matmul(Rx_codebook.conj().transpose(0,1),noise)
        return y_s, y_n
    
    def get_Tx_codebook(self) -> np.ndarray:
        return self.Tx_codebook.numpy()  

    def get_Rx_codebook(self) -> np.ndarray:
        return self.Rx_codebook.numpy()   

class Joint_Tx_Rx_Analog_Beamformer_DFT_Rx(Module):
    def __init__(self, num_antenna_Tx: int, num_antenna_Rx: int, num_beam_Tx: int, num_beam_Rx: int, noise_power: float, norm_factor: float=1.0) -> None:
        super(Joint_Tx_Rx_Analog_Beamformer_DFT_Rx, self).__init__()
        self.num_antenna_Tx = num_antenna_Tx
        self.num_antenna_Rx = num_antenna_Rx
        self.num_beam_Tx = num_beam_Tx
        self.num_beam_Rx = num_beam_Rx
        self.register_buffer('scale_Tx',torch.sqrt(torch.tensor([num_antenna_Tx]).float()))
        self.register_buffer('scale_Rx',torch.sqrt(torch.tensor([num_antenna_Rx]).float()))
        self.register_buffer('noise_power',torch.tensor([noise_power]).float())
        self.register_buffer('norm_factor',torch.tensor([norm_factor]).float())
        self.Tx_codebook_real = Parameter(torch.Tensor(self.num_antenna_Tx, self.num_beam_Tx)) 
        self.Tx_codebook_imag = Parameter(torch.Tensor(self.num_antenna_Tx, self.num_beam_Tx))
        n_ant_per_dim_Rx = int(np.sqrt(num_antenna_Rx))
        n_beam_per_dim_Rx = int(np.sqrt(num_beam_Rx))
        self.register_buffer('Rx_codebook',torch.from_numpy(UPA_DFT_codebook(n_azimuth=n_beam_per_dim_Rx,n_elevation=n_beam_per_dim_Rx,n_antenna_azimuth=n_ant_per_dim_Rx,n_antenna_elevation=n_ant_per_dim_Rx,spacing=0.5).T).to(torch.cfloat))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.Tx_codebook_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.Tx_codebook_imag, a=math.sqrt(5))    
        
    def forward(self, h: Tensor) -> Tensor:
        # h is n_batch x num_antenna_Rx x num_antenna_Tx
        Tx_codebook = self.compute_Tx_codebook() # num_antenna_Tx x num_beam_Tx
        Rx_codebook = self.Rx_codebook # num_antenna_Rx x num_beam_Rx
        y = torch.matmul(h,Tx_codebook) # n_batch x num_antenna_Tx x num_beam_Tx       
        noise_real = torch.normal(0,1, size=y.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise_imag = torch.normal(0,1, size=y.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise = torch.complex(noise_real, noise_imag)
        y_s = torch.matmul(Rx_codebook.conj().transpose(0,1),y)
        y_n = torch.matmul(Rx_codebook.conj().transpose(0,1),noise)
        return y_s, y_n
    
    def compute_Tx_codebook(self) -> Tensor:
        Tx_codebook = torch.complex(self.Tx_codebook_real,self.Tx_codebook_imag)
        Tx_codebook_normalized = torch.div(Tx_codebook,torch.abs(Tx_codebook))/self.scale_Tx
        return Tx_codebook_normalized
    
    def get_Tx_codebook(self) -> np.ndarray:
        with torch.no_grad():
            Tx_codebook = self.compute_Tx_codebook().clone().detach().numpy()
            return Tx_codebook

    def get_Rx_codebook(self) -> np.ndarray:
        return self.Rx_codebook.numpy()  
    
class Joint_Tx_Rx_Analog_Beamformer(Module):
    def __init__(self, num_antenna_Tx: int, num_antenna_Rx: int, num_beam_Tx: int, num_beam_Rx: int, noise_power: float, norm_factor: float=1.0) -> None:
        super(Joint_Tx_Rx_Analog_Beamformer, self).__init__()
        self.num_antenna_Tx = num_antenna_Tx
        self.num_antenna_Rx = num_antenna_Rx
        self.num_beam_Tx = num_beam_Tx
        self.num_beam_Rx = num_beam_Rx
        self.register_buffer('scale_Tx',torch.sqrt(torch.tensor([num_antenna_Tx]).float()))
        self.register_buffer('scale_Rx',torch.sqrt(torch.tensor([num_antenna_Rx]).float()))
        self.register_buffer('noise_power',torch.tensor([noise_power]).float())
        self.register_buffer('norm_factor',torch.tensor([norm_factor]).float())
        self.Tx_codebook_real = Parameter(torch.Tensor(self.num_antenna_Tx, self.num_beam_Tx)) 
        self.Tx_codebook_imag = Parameter(torch.Tensor(self.num_antenna_Tx, self.num_beam_Tx)) 
        self.Rx_codebook_real = Parameter(torch.Tensor(self.num_antenna_Rx, self.num_beam_Rx)) 
        self.Rx_codebook_imag = Parameter(torch.Tensor(self.num_antenna_Rx, self.num_beam_Rx)) 
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.Tx_codebook_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.Tx_codebook_imag, a=math.sqrt(5))
        init.kaiming_uniform_(self.Rx_codebook_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.Rx_codebook_imag, a=math.sqrt(5))        
        
    def forward(self, h: Tensor) -> Tensor:
        # h is n_batch x num_antenna_Rx x num_antenna_Tx
        Tx_codebook = self.compute_Tx_codebook() # num_antenna_Tx x num_beam_Tx
        Rx_codebook = self.compute_Rx_codebook() # num_antenna_Rx x num_beam_Rx
        y = torch.matmul(h,Tx_codebook) # n_batch x num_antenna_Tx x num_beam_Tx       
        noise_real = torch.normal(0,1, size=y.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise_imag = torch.normal(0,1, size=y.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise = torch.complex(noise_real, noise_imag)
        y_s = torch.matmul(Rx_codebook.conj().transpose(0,1),y)
        y_n = torch.matmul(Rx_codebook.conj().transpose(0,1),noise)
        return y_s, y_n
    
    def compute_Tx_codebook(self) -> Tensor:
        Tx_codebook = torch.complex(self.Tx_codebook_real,self.Tx_codebook_imag)
        Tx_codebook_normalized = torch.div(Tx_codebook,torch.abs(Tx_codebook))/self.scale_Tx
        return Tx_codebook_normalized
    
    def compute_Rx_codebook(self) -> Tensor:
        Rx_codebook = torch.complex(self.Rx_codebook_real,self.Rx_codebook_imag)
        Rx_codebook_normalized = torch.div(Rx_codebook,torch.abs(Rx_codebook))/self.scale_Rx
        return Rx_codebook_normalized
    
    def get_Tx_codebook(self) -> np.ndarray:
        with torch.no_grad():
            Tx_codebook = self.compute_Tx_codebook().clone().detach().numpy()
            return Tx_codebook

    def get_Rx_codebook(self) -> np.ndarray:
        with torch.no_grad():
            Rx_codebook = self.compute_Rx_codebook().clone().detach().numpy()
            return Rx_codebook
    
class Beam_Predictor_MLP(nn.Module):
    def __init__(self, num_antenna: int):    
        super(Beam_Predictor_MLP, self).__init__()
        self.num_antenna = num_antenna
        self.register_buffer('scale',torch.sqrt(torch.tensor([num_antenna]).float()))
        self.dense1 = nn.LazyLinear(out_features=num_antenna*5)
        self.dense2 = nn.LazyLinear(out_features=num_antenna*5)
        self.dense3 = nn.LazyLinear(out_features=num_antenna*2)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.dense1(x))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        v_real = out[:,:self.num_antenna]
        v_imag = out[:,self.num_antenna:]
        v = torch.complex(v_real,v_imag)
        v = torch.div(v,torch.abs(v))/self.scale
        return v 
    
class Beam_Predictor_CNN(nn.Module):
    def __init__(self, num_antenna: int):    
        super(Beam_Predictor_CNN, self).__init__()
        self.num_antenna = num_antenna
        self.register_buffer('scale',torch.sqrt(torch.tensor([num_antenna]).float()))
        self.conv1 = nn.LazyConv1d(out_channels=64,kernel_size=3)
        self.conv2 = nn.LazyConv1d(out_channels=64,kernel_size=3)
        self.conv3 = nn.LazyConv1d(out_channels=64,kernel_size=3)
        self.dense1 = nn.LazyLinear(out_features=num_antenna*2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.pool(self.relu(self.conv3(out)))
        out = torch.flatten(out, start_dim=1, end_dim=-1) # flatten except batch dim
        out = self.dense1(out)
        v_real = out[:,:self.num_antenna]
        v_imag = out[:,self.num_antenna:]
        v = torch.complex(v_real,v_imag)
        v = torch.div(v,torch.abs(v))/self.scale
        return v   
    
class Joint_BF_Autoencoder(nn.Module):
    def __init__(self, num_antenna_Tx: int, num_antenna_Rx: int, num_probing_beam_Tx: int, num_probing_beam_Rx: int, 
                 theta_Tx = None, theta_Rx = None,
                 noise_power = 0.0, norm_factor = 1.0, feedback='diagonal', num_feedback = None, 
                 learned_probing = 'TxRx', beam_synthesizer = 'MLP'):
        super(Joint_BF_Autoencoder, self).__init__()        
        self.num_antenna_Tx = num_antenna_Tx
        self.num_antenna_Rx = num_antenna_Rx
        self.num_probing_beam_Tx = num_probing_beam_Tx
        self.num_probing_beam_Rx = num_probing_beam_Rx
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        self.feedback = feedback # 'diagonal','max' or 'full'
        self.num_feedback = num_feedback
        self.learned_probing = learned_probing # 'TxRx','Tx' or 'DFT'
        self.beam_synthesizer = beam_synthesizer # 'MLP' or 'CNN'
        if learned_probing == 'TxRx':
            self.joint_beamformer = Joint_Tx_Rx_Analog_Beamformer(num_antenna_Tx = num_antenna_Tx, num_antenna_Rx = num_antenna_Rx, 
                                                                  num_beam_Tx = num_probing_beam_Tx, num_beam_Rx = num_probing_beam_Rx,
                                                                 noise_power = self.noise_power, norm_factor = self.norm_factor)
        elif learned_probing == 'Tx':
            self.joint_beamformer = Joint_Tx_Rx_Analog_Beamformer_DFT_Rx(num_antenna_Tx = num_antenna_Tx, num_antenna_Rx = num_antenna_Rx, 
                                                                  num_beam_Tx = num_probing_beam_Tx, num_beam_Rx = num_probing_beam_Rx,
                                                                 noise_power = self.noise_power, norm_factor = self.norm_factor)            
        else:
            self.joint_beamformer = Joint_Tx_Rx_Analog_Beamformer_DFT(num_antenna_Tx = num_antenna_Tx, num_antenna_Rx = num_antenna_Rx, 
                                                                  num_beam_Tx = num_probing_beam_Tx, num_beam_Rx = num_probing_beam_Rx,
                                                                 noise_power = self.noise_power, norm_factor = self.norm_factor)            
        assert self.feedback in ['diagonal','max','full']
        if self.feedback == 'diagonal':
            assert self.num_probing_beam_Tx == self.num_probing_beam_Rx, f"number of Tx and Rx probing beams must be the same to use diagonal measurements, got: {self.num_probing_beam_Tx} x {self.num_probing_beam_Rx}"

        if beam_synthesizer == 'MLP':
            self.Tx_beam_predictor = Beam_Predictor_MLP(num_antenna = num_antenna_Tx)
            self.Rx_beam_predictor = Beam_Predictor_MLP(num_antenna = num_antenna_Rx)     
        else: 
            # CNN
            self.Tx_beam_predictor = Beam_Predictor_CNN(num_antenna = num_antenna_Tx)
            self.Rx_beam_predictor = Beam_Predictor_CNN(num_antenna = num_antenna_Rx)                          
            
    def forward(self, x):
        bf_signal_s, bf_signal_n = self.joint_beamformer(x) # n_batch x num_beam_Rx x num_beam_Tx, signal and noise components    
        bf_signal = bf_signal_s + bf_signal_n
        bf_signal_power = torch.pow(torch.abs(bf_signal),2)
        bf_signal_power_noiseless = torch.pow(torch.abs(bf_signal_s),2)
        if self.feedback == 'diagonal':
            # use diagonal elements of Y
            bf_signal_power_feedback = torch.diagonal(bf_signal_power,dim1=1,dim2=2) # n_batch x num_beam_Tx(or num_beam_Rx)
            bf_signal_power_feedback_noiseless = torch.diagonal(bf_signal_power_noiseless,dim1=1,dim2=2) # n_batch x num_beam_Tx(or num_beam_Rx)
            bf_signal_power_measured = bf_signal_power_feedback
        elif self.feedback == 'max':
            # 'max': use max Rx measurement for each Tx beam
            bf_signal_power_feedback,max_ind = bf_signal_power.max(dim=1) # n_batch x num_beam_Tx
            bf_signal_power_feedback_noiseless,max_ind = bf_signal_power_noiseless.max(dim=1) # n_batch x num_beam_Tx
            bf_signal_power_measured = torch.flatten(bf_signal_power,start_dim=1) # n_batch  x (num_beam_Rx * num_beam_Tx)
        else:
            # 'full': use all combinations of beam pair measurements
            bf_signal_power_feedback = torch.flatten(bf_signal_power,start_dim=1) # n_batch  x (num_beam_Rx * num_beam_Tx)
            bf_signal_power_feedback_noiseless = torch.flatten(bf_signal_power_noiseless,start_dim=1) # n_batch x (num_beam_Rx * num_beam_Tx)
            bf_signal_power_measured = bf_signal_power_feedback
        if not self.num_feedback is None:
            sort, indices = torch.sort(bf_signal_power_feedback,dim=1,descending=True)
            feedback_mask = bf_signal_power_feedback >= sort[:,[self.num_feedback]]
            bf_signal_power_feedback = bf_signal_power_feedback*feedback_mask.int().float()
            bf_signal_power_feedback_noiseless = bf_signal_power_feedback_noiseless*feedback_mask.int().float()
        if self.beam_synthesizer == 'CNN':
            bf_signal_power_feedback = bf_signal_power_feedback.unsqueeze(1)
            bf_signal_power_measured = bf_signal_power_measured.unsqueeze(1)
        Tx_beam = self.Tx_beam_predictor(bf_signal_power_feedback)
        Rx_beam = self.Rx_beam_predictor(bf_signal_power_measured)
        return Tx_beam, Rx_beam, bf_signal_power_feedback_noiseless.squeeze()
         
    def get_probing_codebooks(self):
        Tx_probing_codebook = self.joint_beamformer.get_Tx_codebook()
        Rx_probing_codebook = self.joint_beamformer.get_Rx_codebook()
        return Tx_probing_codebook,Rx_probing_codebook   
    
class combined_BF_IA_loss(nn.Module):
    def __init__(self,scale,gamma=1.0,snr_threshold=-5,noise_power_dBm=-94,Tx_power_dBm=30,normalize=True):    
        super(combined_BF_IA_loss, self).__init__()
        self.scale = scale         # normalization factor for h
        self.gamma = gamma
        self.snr_threshold = snr_threshold
        self.Tx_power_dBm = Tx_power_dBm
        self.noise_power_dBm = noise_power_dBm
        self.normalize = normalize
        
    def forward(self,h,tx_beam,rx_beam,probing_bf_power):
        # h: n_batch x num_antenna_Rx x num_antenna_Tx
        # tx_beam: n_batch x num_antenna_Tx
        # rx_beam: n_batch x num_antenna_Rx
        # probing_bf_power: # n_batch x num_beam_Tx x num_beam_Rx (diagonal=False) or num_batch x num_beam (diagonal=True)
        
        tx_beam = tx_beam.unsqueeze(dim=-1)
        rx_beam = rx_beam.unsqueeze(dim=-1)
        y = torch.matmul(rx_beam.conj().transpose(1,2),torch.matmul(h,tx_beam)) # n_batch x 1 x 1
        bf_gain = torch.pow(torch.abs(y.squeeze()),2)
        sq_norm = torch.pow(torch.abs(h.squeeze()),2).sum(dim=(1,2))
        if self.normalize:
            bf_gain = torch.div(bf_gain,sq_norm)
        data_bf_loss = -bf_gain.mean()

        IA_bf_power = torch.amax(probing_bf_power,dim=-1)
        IA_snr = 10*torch.log10(IA_bf_power*self.scale**2)
        IA_snr = self.Tx_power_dBm + IA_snr - self.noise_power_dBm
        misdetected_UE = IA_snr<self.snr_threshold
        misdetection_num = torch.nonzero(misdetected_UE).shape[0]
        
        if misdetected_UE.any():
            IA_bf_gain = torch.div(IA_bf_power,sq_norm)
            misdetection_loss = -IA_bf_gain[misdetected_UE].mean()
            combined_loss = self.gamma*data_bf_loss+(1-self.gamma)*misdetection_loss    
        else:    
            misdetection_loss = torch.tensor([0])
            combined_loss = data_bf_loss
        return combined_loss,data_bf_loss,misdetection_loss,misdetection_num

class BF_loss(nn.Module):
    def __init__(self,noise_power_dBm=-94,Tx_power_dBm=30):    
        super(BF_loss, self).__init__()
        self.Tx_power_dBm = Tx_power_dBm
        self.noise_power_dBm = noise_power_dBm
        
    def forward(self,h,tx_beam,rx_beam):
        # h: n_batch x num_antenna_Rx x num_antenna_Tx
        # tx_beam: n_batch x num_antenna_Tx
        # rx_beam: n_batch x num_antenna_Rx
        
        tx_beam = tx_beam.unsqueeze(dim=-1)
        rx_beam = rx_beam.unsqueeze(dim=-1)
        y = torch.matmul(rx_beam.conj().transpose(1,2),torch.matmul(h,tx_beam)) # n_batch x 1 x 1
        bf_gain = torch.pow(torch.abs(y.squeeze()),2)
        data_bf_loss = -bf_gain.mean()
        return data_bf_loss

class spectral_efficiency_loss(nn.Module):
    def __init__(self,scale,noise_power_dBm=-94,Tx_power_dBm=30):    
        super(spectral_efficiency_loss, self).__init__()
        self.scale = scale         # normalization factor for h
        self.Tx_power_dBm = Tx_power_dBm
        self.noise_power_dBm = noise_power_dBm
        
    def forward(self,h,tx_beam,rx_beam):
        # h: n_batch x num_antenna_Rx x num_antenna_Tx
        # tx_beam: n_batch x num_antenna_Tx
        # rx_beam: n_batch x num_antenna_Rx
        
        tx_beam = tx_beam.unsqueeze(dim=-1)
        rx_beam = rx_beam.unsqueeze(dim=-1)
        y = torch.matmul(rx_beam.conj().transpose(1,2),torch.matmul(h*self.scale,tx_beam)) # n_batch x 1 x 1
        bf_gain = torch.pow(torch.abs(y.squeeze()),2)
        snr = self.Tx_power_dBm + 10*torch.log10(bf_gain) - self.noise_power_dBm
        snr = 10**(snr/10)
        rate = torch.log2(1+snr)
        return -rate.mean()
    
def fit_alt(model, train_loader, val_loader, opt, loss_fn, EPOCHS, model_savefname, loss = 'SPE_loss', device=torch.device('cpu')):
    optimizer = opt
    train_combined_loss_hist,val_combined_loss_hist = [],[]
    
    for epoch in range(EPOCHS):
        model.train()
        train_combined_loss = 0
        for batch_idx, [X_batch] in enumerate(train_loader):
            # var_X_batch = X_batch.to(device)
            var_X_batch = X_batch
            optimizer.zero_grad()
            tx_beam_pred, rx_beam_pred, probing_measured_power = model(var_X_batch)
            combined_loss= loss_fn(var_X_batch, tx_beam_pred, rx_beam_pred)
            combined_loss.backward()
            optimizer.step()
            train_combined_loss += combined_loss.cpu().detach().item()*X_batch.shape[0]
        train_combined_loss /= len(train_loader.dataset)
        train_combined_loss_hist.append(train_combined_loss)

        if epoch % 100 == 0:
            model.eval()
            val_combined_loss = 0
            for batch_idx, [X_batch] in enumerate(val_loader):
                # var_X_batch = X_batch.to(device)
                var_X_batch = X_batch
                with torch.no_grad():
                    tx_beam_pred, rx_beam_pred, probing_measured_power = model(var_X_batch)
                    combined_loss = loss_fn(var_X_batch, tx_beam_pred, rx_beam_pred)
                val_combined_loss += combined_loss.cpu().detach().item()*X_batch.shape[0]    
            val_combined_loss /= len(val_loader.dataset)
            val_combined_loss_hist.append(val_combined_loss)
            if loss == 'SPE_loss':
                print(('Epoch : {}, SPE (bps/Hz): Training = {:.2f}, Validation = {:.2f}.').format(epoch,train_combined_loss,val_combined_loss))
            else:
                print(('Epoch : {}, BF loss: Training = {:.2f}, Validation = {:.2f}.').format(epoch,train_combined_loss,val_combined_loss))
            torch.save(model.state_dict(),model_savefname+"_epoch_{}.pt".format(epoch))
        combined_loss_hist = (train_combined_loss_hist, val_combined_loss_hist)
    return combined_loss_hist
    
def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS, model_savefname, h_NMSE_dB = -np.inf, device=torch.device('cpu')):
    optimizer = opt
    train_combined_loss_hist,val_combined_loss_hist = [],[]
    train_bf_loss_hist,val_bf_loss_hist = [],[]
    train_misdetection_loss_hist,val_misdetection_loss_hist = [],[]
    train_misdetection_rate_hist,val_misdetection_rate_hist = [],[]
    h_NMSE = 10**(h_NMSE_dB/10)
    
    for epoch in range(EPOCHS):
        model.train()
        train_combined_loss, train_bf_loss, train_misdetection_loss, train_misdetection_rate = 0,0,0,0
        for batch_idx, [X_batch] in enumerate(train_loader):
            if h_NMSE>0:
                h_pow = torch.pow(torch.linalg.norm(X_batch,dim=(1,2)),2)
                error_pow = h_pow*h_NMSE
                error_pow_per_antenna = error_pow/X_batch.shape[1]/X_batch.shape[2]
                error_pow_per_antenna = error_pow_per_antenna.reshape((X_batch.shape[0],1,1)).tile((1,X_batch.shape[1],X_batch.shape[2]))
                h_error_complex = torch.randn(X_batch.shape,dtype=torch.cfloat).to(error_pow_per_antenna.device)*torch.sqrt(error_pow_per_antenna)
                X_batch = X_batch + h_error_complex
            # var_X_batch = X_batch.to(device)
            var_X_batch = X_batch
            optimizer.zero_grad()
            tx_beam_pred, rx_beam_pred, probing_measured_power = model(var_X_batch)
            combined_loss,data_bf_loss,misdetection_loss,misdetection_num = loss_fn(var_X_batch, tx_beam_pred, rx_beam_pred, probing_measured_power)
            combined_loss.backward()
            optimizer.step()
            train_combined_loss += combined_loss.cpu().detach().item()*X_batch.shape[0]
            train_bf_loss += data_bf_loss.cpu().detach().item()*X_batch.shape[0]
            train_misdetection_loss += misdetection_loss.cpu().detach().item()*X_batch.shape[0]
            train_misdetection_rate += misdetection_num
            
        train_combined_loss /= len(train_loader.dataset)
        train_bf_loss /= len(train_loader.dataset)
        train_misdetection_loss /= len(train_loader.dataset)
        train_misdetection_rate /= len(train_loader.dataset)/100

        model.eval()
        val_combined_loss, val_bf_loss, val_misdetection_loss, val_misdetection_rate = 0,0,0,0
        for batch_idx, [X_batch] in enumerate(val_loader):
            # var_X_batch = X_batch.to(device)
            var_X_batch = X_batch
            with torch.no_grad():
                tx_beam_pred, rx_beam_pred, probing_measured_power = model(var_X_batch)
                combined_loss,data_bf_loss,misdetection_loss,misdetection_num = loss_fn(var_X_batch, tx_beam_pred, rx_beam_pred, probing_measured_power)
            val_combined_loss += combined_loss.cpu().detach().item()*X_batch.shape[0]
            val_bf_loss += data_bf_loss.cpu().detach().item()*X_batch.shape[0]
            val_misdetection_loss += misdetection_loss.cpu().detach().item()*X_batch.shape[0]
            val_misdetection_rate += misdetection_num
            
        val_combined_loss /= len(val_loader.dataset)
        val_bf_loss /= len(val_loader.dataset)
        val_misdetection_loss /= len(val_loader.dataset)
        val_misdetection_rate /= len(val_loader.dataset)/100
        
        train_combined_loss_hist.append(train_combined_loss)
        train_bf_loss_hist.append(train_bf_loss)
        train_misdetection_loss_hist.append(train_misdetection_loss)
        train_misdetection_rate_hist.append(train_misdetection_rate)
        
        val_combined_loss_hist.append(val_combined_loss)
        val_bf_loss_hist.append(val_bf_loss)
        val_misdetection_loss_hist.append(val_misdetection_loss)
        val_misdetection_rate_hist.append(val_misdetection_rate)

        if epoch % 100 == 0:
            print(('Epoch : {}, Combined loss: Training = {:.2f}, Validation = {:.2f}; '
                   'BF loss: Training = {:.2f}, Validation = {:.2f}; '
                   'Misdetection loss: Training = {:.2f}, Validation = {:.2f}; '
                   'Misdetection rate: Training = {:.2f}, Validation = {:.2f}.').format(epoch,train_combined_loss,val_combined_loss,
                                                                                       train_bf_loss,val_bf_loss,
                                                                                       train_misdetection_loss,val_misdetection_loss,
                                                                                       train_misdetection_rate,val_misdetection_rate))
            torch.save(model.state_dict(),model_savefname+"_epoch_{}.pt".format(epoch))
        combined_loss_hist = (train_combined_loss_hist, val_combined_loss_hist)
        bf_loss_hist = (train_bf_loss_hist, val_bf_loss_hist)
        misdetection_loss_hist = (train_misdetection_loss_hist, val_misdetection_loss_hist)
        misdetection_rate_hist = (train_misdetection_rate_hist, val_misdetection_rate_hist)
        
    return combined_loss_hist, bf_loss_hist, misdetection_loss_hist, misdetection_rate_hist

class Beam_Classifier(nn.Module):
    def __init__(self, num_antenna: int, num_beams: int):    
        super(Beam_Classifier, self).__init__()
        self.num_antenna = num_antenna
        self.num_beams = num_beams
        self.register_buffer('scale',torch.sqrt(torch.tensor([num_antenna]).float()))
        self.dense1 = nn.LazyLinear(out_features=num_antenna*5)
        self.dense2 = nn.LazyLinear(out_features=num_antenna*5)
        self.dense3 = nn.LazyLinear(out_features=num_beams)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.dense1(x))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out 
    
class Joint_BF_Classifier(nn.Module):
    def __init__(self, num_antenna_Tx: int, num_antenna_Rx: int, num_probing_beam_Tx: int, num_probing_beam_Rx: int, 
                 num_Tx_beams: int, num_Rx_beams: int,
                 noise_power = 0.0, norm_factor = 1.0):
        super(Joint_BF_Classifier, self).__init__()        
        self.num_antenna_Tx = num_antenna_Tx
        self.num_antenna_Rx = num_antenna_Rx
        self.num_probing_beam_Tx = num_probing_beam_Tx
        self.num_probing_beam_Rx = num_probing_beam_Rx
        self.num_beam_Tx = num_Tx_beams
        self.num_beam_Rx = num_Rx_beams
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        self.joint_beamformer = Joint_Tx_Rx_Analog_Beamformer(num_antenna_Tx = num_antenna_Tx, num_antenna_Rx = num_antenna_Rx, 
                                                              num_beam_Tx = num_probing_beam_Tx, num_beam_Rx = num_probing_beam_Rx,
                                                             noise_power = self.noise_power, norm_factor = self.norm_factor)
        
        assert self.num_probing_beam_Tx == self.num_probing_beam_Rx, f"number of Tx and Rx probing beams must be the same to use diagonal measurements, got: {self.num_probing_beam_Tx} x {self.num_probing_beam_Rx}"
        self.Tx_beam_predictor = Beam_Classifier(num_antenna = num_antenna_Tx, num_beams = num_Tx_beams)
        self.Rx_beam_predictor = Beam_Classifier(num_antenna = num_antenna_Rx, num_beams = num_Rx_beams)                     
         
    def forward(self, x):
        bf_signal_s, bf_signal_n = self.joint_beamformer(x) # n_batch x num_beam_Rx x num_beam_Tx, signal and noise components   
        bf_signal = bf_signal_s + bf_signal_n
        bf_signal_power = torch.pow(torch.abs(bf_signal),2)      
        bf_signal_power = torch.diagonal(bf_signal_power,dim1=1,dim2=2) # n_batch x num_beam_Tx(or num_beam_Rx)
        # bf_signal_power_noiseless = torch.pow(torch.abs(bf_signal_s),2)
        # bf_signal_power_feedback_noiseless = torch.diagonal(bf_signal_power_noiseless,dim1=1,dim2=2) # n_batch x num_beam_Tx(or num_beam_Rx)
        Tx_beam = self.Tx_beam_predictor(bf_signal_power)
        Rx_beam = self.Rx_beam_predictor(bf_signal_power)
        return Tx_beam, Rx_beam
         
    def get_probing_codebooks(self):
        Tx_probing_codebook = self.joint_beamformer.get_Tx_codebook()
        Rx_probing_codebook = self.joint_beamformer.get_Rx_codebook()
        return Tx_probing_codebook,Rx_probing_codebook   
    

def fit_CB(model, train_loader, val_loader, opt, loss_fn, EPOCHS, model_savefname, h_val, h_NMSE_dB = -np.inf, device=torch.device("cpu")):
    
    DFT_codebook_TX = UPA_DFT_codebook(n_azimuth=8*2,n_elevation=8*2,n_antenna_azimuth=8,n_antenna_elevation=8,spacing=0.5).T
    DFT_codebook_RX = UPA_DFT_codebook(n_azimuth=4*2,n_elevation=4*2,n_antenna_azimuth=4,n_antenna_elevation=4,spacing=0.5).T

    DFT_codebook_TX_torch = torch.from_numpy(DFT_codebook_TX).cfloat().to(device)
    DFT_codebook_RX_torch = torch.from_numpy(DFT_codebook_RX).cfloat().to(device)
    
    optimizer = opt
    train_loss_hist, val_loss_hist = np.zeros((2,EPOCHS)), np.zeros((2,EPOCHS))
    train_acc_hist, val_acc_hist = np.zeros((2,EPOCHS)), np.zeros((2,EPOCHS))
    h_NMSE = 10**(h_NMSE_dB/10)
    for epoch in range(EPOCHS):
        model.train()
        train_loss_tx, train_loss_rx = 0,0
        train_acc_tx, train_acc_rx = 0,0
        for batch_idx, (h_batch, tx_label_batch, rx_label_batch) in enumerate(train_loader):
            if h_NMSE>0:
                h_pow = torch.pow(torch.linalg.norm(h_batch,dim=(1,2)),2)
                error_pow = h_pow*h_NMSE
                error_pow_per_antenna = error_pow/h_batch.shape[1]/h_batch.shape[2]
                error_pow_per_antenna = error_pow_per_antenna.reshape((h_batch.shape[0],1,1)).tile((1,h_batch.shape[1],h_batch.shape[2]))
                h_error_complex = torch.randn(h_batch.shape,dtype=torch.cfloat).to(error_pow_per_antenna.device)*torch.sqrt(error_pow_per_antenna)
                h_batch = h_batch + h_error_complex
            # with torch.no_grad():
                dft_bf_gain_batch = torch.abs(DFT_codebook_RX_torch.conj().transpose(0,1) @ h_batch @ DFT_codebook_TX_torch)**2
                noisy_rx_label_batch, noisy_tx_label_batch = unravel_index(dft_bf_gain_batch.reshape(dft_bf_gain_batch.shape[0],-1).argmax(1),dft_bf_gain_batch[0].shape)
                tx_label_batch = noisy_tx_label_batch
                rx_label_batch = noisy_rx_label_batch
            optimizer.zero_grad()
            tx_pred,rx_pred = model(h_batch)
            tx_loss = loss_fn(tx_pred, tx_label_batch)
            rx_loss = loss_fn(rx_pred, rx_label_batch)
            loss = tx_loss + rx_loss
            loss.backward()
            optimizer.step()
            train_loss_tx += tx_loss.detach().item()*h_batch.shape[0]
            train_loss_rx += rx_loss.detach().item()*h_batch.shape[0]
            train_acc_tx += (tx_pred.argmax(dim=1) == tx_label_batch).sum().item()
            train_acc_rx += (rx_pred.argmax(dim=1) == rx_label_batch).sum().item()
        train_loss_tx /= len(train_loader.dataset)
        train_loss_rx /= len(train_loader.dataset)
        train_acc_tx /= len(train_loader.dataset)
        train_acc_rx /= len(train_loader.dataset)

        train_loss_hist[0,epoch] = train_loss_tx
        train_loss_hist[1,epoch] = train_loss_rx
        train_acc_hist[0,epoch] = train_acc_tx
        train_acc_hist[1,epoch] = train_acc_rx
        
        model.eval()
        val_loss_tx, val_loss_rx = 0,0
        val_acc_tx, val_acc_rx = 0,0
        for batch_idx, (h_batch, tx_label_batch, rx_label_batch) in enumerate(val_loader):
            with torch.no_grad():
                tx_pred,rx_pred = model(h_batch)
                tx_loss = loss_fn(tx_pred, tx_label_batch)
                rx_loss = loss_fn(rx_pred, rx_label_batch)
                val_loss_tx += tx_loss.detach().item()*h_batch.shape[0]
                val_loss_rx += rx_loss.detach().item()*h_batch.shape[0]
                val_acc_tx += (tx_pred.argmax(dim=1) == tx_label_batch).sum().item()
                val_acc_rx += (rx_pred.argmax(dim=1) == rx_label_batch).sum().item()
        val_loss_tx /= len(val_loader.dataset)
        val_loss_rx /= len(val_loader.dataset)
        val_acc_tx /= len(val_loader.dataset)
        val_acc_rx /= len(val_loader.dataset)
        
        val_loss_hist[0,epoch] = val_loss_tx
        val_loss_hist[1,epoch] = val_loss_rx
        val_acc_hist[0,epoch] = val_acc_tx
        val_acc_hist[1,epoch] = val_acc_rx     
        
        if epoch % 100 == 0:
            tx_beams_pred = tx_pred.cpu().detach().numpy().argmax(axis=-1)
            rx_beams_pred = rx_pred.cpu().detach().numpy().argmax(axis=-1)
            tx_beams_pred = np.expand_dims(DFT_codebook_TX[:,tx_beams_pred].T,axis=-1) # num_ant_Tx x val_size -> val_size x num_ant_Tx x 1
            rx_beams_pred =  np.expand_dims(DFT_codebook_RX[:,rx_beams_pred].T,axis=-1) # num_ant_Rx x val_size -> val_size x num_ant_Rx x 1
            bf_gain_pred = abs(np.transpose(rx_beams_pred.conj(),(0,2,1)) @ h_val @ tx_beams_pred)**2
            bf_gain_normalized = np.linalg.norm(h_val,axis=(1,2))**2
            bf_gain_normalized = bf_gain_pred.squeeze()/bf_gain_normalized
            print('Epoch : {}, Training loss Tx = {:.2f}, Training Acc Tx = {:.2f}, Training loss Rx = {:.2f}, Training Acc Rx = {:.2f}, Val loss Tx = {:.2f}, Val Acc Tx = {:.2f}, Val loss Rx = {:.2f}, Val Acc Rx = {:.2f}, Val BF gain = {:.4f}.'.format(epoch,train_loss_tx,train_acc_tx,train_loss_rx,train_acc_rx,val_loss_tx,val_acc_tx,val_loss_rx,val_acc_rx,bf_gain_normalized.mean()))
            # torch.save(model.state_dict(),model_savefname)
            torch.save(model.state_dict(),model_savefname+"_epoch_{}.pt".format(epoch))

    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist

def eval_model(model,torch_x_test,h_test,noise_power,prediction_mode='GF',feedback_mode='diagonal',topk_arr=[1,3]):
    DFT_codebook_TX = UPA_DFT_codebook(n_azimuth=8*2,n_elevation=8*2,n_antenna_azimuth=8,n_antenna_elevation=8,spacing=0.5).T
    DFT_codebook_RX = UPA_DFT_codebook(n_azimuth=4*2,n_elevation=4*2,n_antenna_azimuth=4,n_antenna_elevation=4,spacing=0.5).T
    model.eval()
    with torch.no_grad():
        if prediction_mode=='GF':
            tx_beam_pred, rx_beam_pred, probing_bf_power = model(torch_x_test)
            rx_beam_pred = rx_beam_pred.detach().unsqueeze(dim=-1).numpy() # n_batch x num_antenna_Rx x 1
            tx_beam_pred = tx_beam_pred.detach().unsqueeze(dim=-1).numpy() # n_batch x num_antenna_Tx x 1
            predicted_bf_gain = np.transpose(rx_beam_pred.conj(),axes=(0,2,1)) @ h_test @ tx_beam_pred
            predicted_bf_gain = np.power(np.absolute(predicted_bf_gain),2).squeeze()   
        else: # top-1 and top-3 for CB
            tx_beam_pred, rx_beam_pred = model(torch_x_test)
            predicted_bf_gain = np.zeros((len(topk_arr),h_test.shape[0]))
            topk_max = max(topk_arr)
            tx_beam_pred = np.transpose(DFT_codebook_TX[:,tx_beam_pred.detach().numpy().argsort(axis=-1)[:,-topk_max:]],(1,0,2)) # num_ant_Tx x n_batch -> n_batch x num_ant_Tx x topk_max
            rx_beam_pred = np.transpose(DFT_codebook_RX[:,rx_beam_pred.detach().numpy().argsort(axis=-1)[:,-topk_max:]],(1,0,2)) # num_ant_Rx x n_batch -> n_batch x num_ant_Rx x topk_max
            rx_signal = h_test @ tx_beam_pred
            rx_noise_real = np.random.normal(loc=0,scale=1,size=rx_signal.shape)*np.sqrt(noise_power/2)
            rx_noise_imag = np.random.normal(loc=0,scale=1,size=rx_signal.shape)*np.sqrt(noise_power/2)
            rx_signal_w_noise = rx_signal + rx_noise_real + 1j*rx_noise_imag
            bf_gain_w_noise = np.power(np.absolute(np.transpose(rx_beam_pred.conj(),(0,2,1)) @ rx_signal_w_noise),2)  # val_size x topk_max x topk_max
            bf_gain_wo_noise = np.power(np.absolute(np.transpose(rx_beam_pred.conj(),(0,2,1)) @ rx_signal),2)  # val_size x topk_max x topk_max
            
            for i,topk in enumerate(topk_arr):
                optimal_beam_pair_search = bf_gain_w_noise[:,-topk:,-topk:].reshape(bf_gain_w_noise.shape[0],-1).argmax(axis=-1)
                bf_gain_pred = bf_gain_wo_noise[:,-topk:,-topk:].reshape(bf_gain_wo_noise.shape[0],-1)[np.arange(bf_gain_wo_noise.shape[0]),optimal_beam_pair_search]    
                predicted_bf_gain[i,:] = bf_gain_pred
    
    probing_codebook_Tx,probing_codebook_Rx = model.get_probing_codebooks()
    probing_codebook_bf_gain = np.power(np.absolute(probing_codebook_Rx.conj().T @ h_test @ probing_codebook_Tx),2)
    if feedback_mode == 'diagonal':
        probing_codebook_bf_gain = np.diagonal(probing_codebook_bf_gain,axis1=1,axis2=2) # n_batch x num_beam_Tx
    elif feedback_mode == 'max':
        probing_codebook_bf_gain = probing_codebook_bf_gain.max(axis=1) # n_batch x num_beam_Tx
    else:
        probing_codebook_bf_gain = probing_codebook_bf_gain.reshape(probing_codebook_bf_gain.shape[0],-1) # n_batch x (num_beam_Tx * num_beam_Rx)
    probing_codebook_bf_gain = probing_codebook_bf_gain.max(axis=-1)
    return predicted_bf_gain, probing_codebook_bf_gain