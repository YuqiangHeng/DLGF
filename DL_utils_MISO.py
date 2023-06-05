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

class PhaseShifter(nn.Module):
    def __init__(self, num_antenna, num_beam):
        super(PhaseShifter, self).__init__()
        self.num_antenna = num_antenna
        self.num_beam = num_beam
        # self.W = Parameter(torch.Tensor(self.num_antenna, self.num_beam,dtype=torch.complex64))
        self.W_real = Parameter(torch.Tensor(self.num_antenna, self.num_beam))
        self.W_imag = Parameter(torch.Tensor(self.num_antenna, self.num_beam))
        self.register_buffer('scale',torch.sqrt(torch.tensor([num_antenna]).float()))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_imag, a=math.sqrt(5))
    
    def forward(self,h):
        # h: n_batch  x 1 x num_antenna
        W = self.get_W()
        out = torch.matmul(h,W) #n_batch x 1 x num_beam
        return out
    
    def get_W(self):
        W = torch.complex(self.W_real,self.W_imag)
        W_normalized = torch.div(W,torch.abs(W))/self.scale # element-wise divide by norm for constant modulus
        return W_normalized
        
    def get_codebook(self) -> np.ndarray:
        with torch.no_grad():
            W = self.get_W().clone().detach().numpy()
            return W
    
class BF_Autoencoder(nn.Module):
    def __init__(self, num_antenna, num_probing_beam, noise_power = 0.0, norm_factor = 1.0, mode = 'GF', num_beam = None):
        super(BF_Autoencoder, self).__init__()        
        self.num_antenna = num_antenna
        self.num_probing_beam = num_probing_beam
        self.register_buffer('norm_factor',torch.tensor([norm_factor]).float())
        self.register_buffer('noise_power',torch.tensor([noise_power]).float())
        self.mode = mode
        self.num_beam = num_beam
        self.beamformer = PhaseShifter(num_antenna=self.num_antenna, num_beam=self.num_probing_beam)
        if self.mode == 'GF':
            self.beam_predictor = Beam_Synthesizer(num_antenna = self.num_antenna)
        else:
            if self.num_beam is None:
               self.num_beam = self.num_antenna
            self.beam_predictor = Beam_Classifier(num_antenna = self.num_antenna, num_beam = self.num_beam) 
        
    def forward(self, h):
        # h: n_batch  x 1 x num_antenna
        bf_signal = self.beamformer(h).squeeze(dim=1) # n_batch x 1 x num_beam -> n_batch x num_beam
        noise_real = torch.normal(0,1, size=bf_signal.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise_imag = torch.normal(0,1, size=bf_signal.size()).to(h.device)*torch.sqrt(self.noise_power/2)/self.norm_factor
        noise = torch.complex(noise_real, noise_imag)
        bf_signal_w_noise = bf_signal + noise
        bf_signal_power = torch.pow(torch.abs(bf_signal_w_noise),2)
        out = self.beam_predictor(bf_signal_power)   
        return out
    
    def get_codebook(self) -> np.ndarray:
        return self.beamformer.get_codebook()
    
class Beam_Synthesizer(nn.Module):
    def __init__(self, num_antenna: int):    
        super(Beam_Synthesizer, self).__init__()
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
    
class Beam_Classifier(nn.Module):
    def __init__(self, num_antenna: int, num_beam: int):    
        super(Beam_Classifier, self).__init__()
        self.num_antenna = num_antenna
        self.num_beam = num_beam
        self.dense1 = nn.LazyLinear(out_features=num_antenna*5)
        self.dense2 = nn.LazyLinear(out_features=num_antenna*5)
        self.dense3 = nn.LazyLinear(out_features=num_beam)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.dense1(x))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out 
    
def BF_gain_loss(h,v):
    # h: n_batch  x 1 x num_antenna
    # v: n_batch x num_antenna
    v = v.unsqueeze(dim=-1)
    bf_signal = torch.matmul(h,v).squeeze()
    bf_power = torch.pow(torch.abs(bf_signal),2)
    norm = torch.pow(torch.abs(h.squeeze()),2).sum(dim=-1)
    bf_gain = torch.div(bf_power,norm)
    return -bf_gain.mean()

class spectral_efficiency_loss(nn.Module):
    def __init__(self,scale,noise_power_dBm=-94,Tx_power_dBm=30):    
        super(spectral_efficiency_loss, self).__init__()
        self.scale = scale         # normalization factor for h
        self.Tx_power_dBm = Tx_power_dBm
        self.noise_power_dBm = noise_power_dBm
        
    def forward(self,h,tx_beam):
        # h: n_batch x 1 x num_antenna_Tx
        # tx_beam: n_batch x num_antenna_Tx
        tx_beam = tx_beam.unsqueeze(dim=-1)
        y = torch.matmul(h*self.scale,tx_beam) # n_batch x 1
        bf_gain = torch.pow(torch.abs(y.squeeze()),2)
        snr = self.Tx_power_dBm + 10*torch.log10(bf_gain) - self.noise_power_dBm
        snr = 10**(snr/10)
        rate = torch.log2(1+snr)
        return -rate.mean()

def fit_GF(model, train_loader, val_loader, opt, loss_fn, EPOCHS, model_savefname):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, [var_X_batch] in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(var_X_batch, output)
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach().item()
        train_loss /= batch_idx + 1
        model.eval()
        val_loss = 0
        for batch_idx, [var_X_batch] in enumerate(val_loader):
            with torch.no_grad():
                output = model(var_X_batch)
                loss = loss_fn(var_X_batch, output)
            val_loss += loss.cpu().detach().item()
        val_loss /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 100 == 0:
            print('Epoch : {}, Training loss = {:.5f}, Validation loss = {:.5f}.'.format(epoch,train_loss,val_loss))
            torch.save(model.state_dict(),model_savefname+"_epoch_{}.pt".format(epoch))
    return train_loss_hist, val_loss_hist

def fit_CB(model, train_loader, val_loader, opt, loss_fn, EPOCHS, model_savefname):
    optimizer = opt
    train_loss_hist, val_loss_hist = np.zeros(EPOCHS), np.zeros(EPOCHS)
    train_acc_hist, val_acc_hist = np.zeros(EPOCHS), np.zeros(EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc = 0,0
        for batch_idx, (h_batch, label_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(h_batch)
            loss = loss_fn(pred, label_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()*h_batch.shape[0]
            train_acc += (pred.argmax(dim=1) == label_batch).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        train_loss_hist[epoch] = train_loss
        train_acc_hist[epoch] = train_acc
        
        model.eval()
        val_loss, val_acc = 0,0
        for batch_idx, (h_batch, label_batch) in enumerate(val_loader):
            with torch.no_grad():
                pred = model(h_batch)
                loss = loss_fn(pred, label_batch)
                val_loss += loss.detach().item()*h_batch.shape[0]
                val_acc += (pred.argmax(dim=1) == label_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        
        val_loss_hist[epoch] = val_loss
        val_acc_hist[epoch] = val_acc
        
        if epoch % 100 == 0:
            print('Epoch : {}, Training loss = {:.2f}, Training Acc = {:.2f}, Val loss = {:.2f}, Val Acc = {:.2f}.'.format(epoch,train_loss,train_acc,val_loss,val_acc))
            torch.save(model.state_dict(),model_savefname+"_epoch_{}.pt".format(epoch))

    return train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist