# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:49:09 2021

@author: ethan
"""
import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import beam_utils

class Hybrid_Beamformer(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        theta: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from uniform(0,2*pi)
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    scale: float

    def __init__(self, n_antenna: int, n_beam: int, n_rf: int, n_stream: int = 1, use_bias: bool = False, scale: float=1, init_criterion = 'xavier_normal') -> None:
        super(Hybrid_Beamformer, self).__init__()
        self.n_antenna = n_antenna
        self.in_dim = self.n_antenna * 2
        self.n_rf = n_rf
        self.n_beam = n_beam
        self.scale = scale
        self.init_criterion = init_criterion
        self.n_stream = n_stream
        self.theta = Parameter(torch.Tensor(self.n_beam, self.n_antenna, self.n_rf)) 
        self.real_kernel = Parameter(torch.Tensor(self.n_beam, self.n_rf, self.n_stream)) 
        self.imag_kernel = Parameter(torch.Tensor(self.n_beam, self.n_rf, self.n_stream)) 
        self.use_bias = use_bias
        self.fb_norm = Complex_Frobenius_Norm((self.n_antenna*2,self.n_stream*2))
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(self.n_beam, self.n_antenna, self.n_stream)) 
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.uniform_(self.theta, a=0, b=2*np.pi)
        self.analog_real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.analog_imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #
        
        if self.init_criterion == 'xavier_normal':
            init.xavier_normal_(self.real_kernel,gain = 1/np.sqrt(2))
            init.xavier_normal_(self.imag_kernel,gain = 1/np.sqrt(2))
        elif self.init_criterion == 'kaiming_normal':
            init.kaiming_normal_(self.real_kernel)
            init.kaiming_normal_(self.imag_kernel)        
        else:
             raise  NotImplementedError 
             
        if self.use_bias:
            if self.init_criterion == 'xavier_normal':
                init.xavier_normal_(self.bias,gain = 1/np.sqrt(2))
            elif self.init_criterion == 'kaiming_normal':
                init.kaiming_normal_(self.bias)
            else:
                 raise  NotImplementedError             
             
    def forward(self, inputs: Tensor) -> Tensor: 
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1).unsqueeze(1) # so that inputs has shape b x 1 x 1 x N_T
        
        cat_kernels_4_real_digital = torch.cat(
            (self.real_kernel, -self.imag_kernel),
            dim=-1
        )
        cat_kernels_4_imag_digital = torch.cat(
            (self.imag_kernel, self.real_kernel),
            dim=-1
        )
        cat_kernels_4_complex_digital = torch.cat(
            (cat_kernels_4_real_digital, cat_kernels_4_imag_digital),
            dim=1
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]

        self.real_kernel_analog = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel_analog = (1 / self.scale) * torch.sin(self.theta)  #        
        cat_kernels_4_real_analog = torch.cat(
            (self.real_kernel_analog, -self.imag_kernel_analog),
            dim=-1
        )
        cat_kernels_4_imag_analog = torch.cat(
            (self.imag_kernel_analog, self.real_kernel_analog),
            dim=-1
        )
        cat_kernels_4_complex_analog = torch.cat(
            (cat_kernels_4_real_analog, cat_kernels_4_imag_analog),
            dim=1
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]        
        cat_kernels_4_complex_hybrid = torch.matmul(cat_kernels_4_complex_analog, cat_kernels_4_complex_digital) # shape n_beam x n_antenna*2 x n_stream*2
        norm_factor = self.fb_norm(cat_kernels_4_complex_hybrid) # shape n_beam vector
        norm_factor = norm_factor.unsqueeze(1).unsqueeze(1)
        norm_factor = norm_factor.repeat(1,self.n_antenna*2,self.n_stream*2)
        # norm_factor = cat_kernels_4_complex_hybrid.sum(dim=0).repeat(1, self.n_antenna*2)
        cat_kernels_4_complex_hybrid_normalized = cat_kernels_4_complex_hybrid * (1/norm_factor)
        if self.use_bias:
            cat_kernels_4_complex_hybrid_normalized = cat_kernels_4_complex_hybrid_normalized + self.bias
        output = torch.matmul(inputs, cat_kernels_4_complex_hybrid_normalized)
        return output.squeeze()
    
    def get_hybrid_weights(self):
        with torch.no_grad():
            cat_kernels_4_real_digital = torch.cat(
                (self.real_kernel, -self.imag_kernel),
                dim=-1
            )
            cat_kernels_4_imag_digital = torch.cat(
                (self.imag_kernel, self.real_kernel),
                dim=-1
            )
            cat_kernels_4_complex_digital = torch.cat(
                (cat_kernels_4_real_digital, cat_kernels_4_imag_digital),
                dim=1
            )  # This block matrix represents the conjugate transpose of the original:
            # [ W_R, -W_I; W_I, W_R]
    
            self.real_kernel_analog = (1 / self.scale) * torch.cos(self.theta)  #
            self.imag_kernel_analog = (1 / self.scale) * torch.sin(self.theta)  #        
            cat_kernels_4_real_analog = torch.cat(
                (self.real_kernel_analog, -self.imag_kernel_analog),
                dim=-1
            )
            cat_kernels_4_imag_analog = torch.cat(
                (self.imag_kernel_analog, self.real_kernel_analog),
                dim=-1
            )
            cat_kernels_4_complex_analog = torch.cat(
                (cat_kernels_4_real_analog, cat_kernels_4_imag_analog),
                dim=1
            )  # This block matrix represents the conjugate transpose of the original:
            # [ W_R, -W_I; W_I, W_R]        
            cat_kernels_4_complex_hybrid = torch.matmul(cat_kernels_4_complex_analog, cat_kernels_4_complex_digital) # shape n_beam x n_antenna*2 x n_stream*2
            norm_factor = self.fb_norm(cat_kernels_4_complex_hybrid) # shape n_beam vector
            norm_factor = norm_factor.unsqueeze(1).unsqueeze(1)
            norm_factor = norm_factor.repeat(1,self.n_antenna*2,self.n_stream*2)
            # norm_factor = cat_kernels_4_complex_hybrid.sum(dim=0).repeat(1, self.n_antenna*2)
            cat_kernels_4_complex_hybrid_normalized = cat_kernels_4_complex_hybrid * (1/norm_factor)  
        cat_kernels_4_complex_hybrid_normalized = cat_kernels_4_complex_hybrid_normalized.detach().numpy()
        hybrid_kernel_real = cat_kernels_4_complex_hybrid_normalized[:,:self.n_antenna,:self.n_stream]
        hybrid_kernel_imag = cat_kernels_4_complex_hybrid_normalized[:,self.n_antenna:,:self.n_stream]
        hybrid_beam_weights = hybrid_kernel_real + 1j*hybrid_kernel_imag
        return hybrid_beam_weights
    

    
class Complex_Dense(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        theta: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from uniform(0,2*pi)
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    scale: float

    def __init__(self, in_features: int, out_features: int, use_bias: bool = False, scale: float=1, init_criterion = 'xavier_normal') -> None:
        super(Complex_Dense, self).__init__()
        self.in_features = in_features
        self.in_dim = self.in_features//2
        self.out_features = out_features
        self.scale = scale
        self.init_criterion = init_criterion
        self.real_kernel = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.imag_kernel = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(self.in_features, self.out_features)) 
        self.reset_parameters()

    def reset_parameters(self) -> None:
        
        if self.init_criterion == 'xavier_normal':
            init.xavier_normal_(self.real_kernel,gain = 1/np.sqrt(2))
            init.xavier_normal_(self.imag_kernel,gain = 1/np.sqrt(2))
        elif self.init_criterion == 'kaiming_normal':
            init.kaiming_normal_(self.real_kernel)
            init.kaiming_normal_(self.imag_kernel)        
        else:
             raise  NotImplementedError      
                        
    def forward(self, inputs: Tensor) -> Tensor:    
        cat_kernels_4_real = torch.cat(
            (self.real_kernel, -self.imag_kernel),
            dim=-1
        )
        cat_kernels_4_imag = torch.cat(
            (self.imag_kernel, self.real_kernel),
            dim=-1
        )
        cat_kernels_4_complex = torch.cat(
            (cat_kernels_4_real, cat_kernels_4_imag),
            dim=0
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]
        # weight_power = torch.pow(self.real_kernel,2) + torch.pow(self.imag_kernel,2)
        # weight_magnitue = torch.sqrt(weight_power)
        # output = F.linear(inputs, cat_kernels_4_complex)
        output = torch.matmul(inputs, (1 / self.scale) * cat_kernels_4_complex)
        if self.use_bias:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    
    def get_weights(self):
        return self.real_kernel, self.imag_kernel
    
    
class PhaseShifter(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        theta: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from uniform(0,2*pi)
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    scale: float
    theta: Tensor

    def __init__(self, in_features: int, out_features: int, scale: float=1, theta = None) -> None:
        super(PhaseShifter, self).__init__()
        self.in_features = in_features
        self.in_dim = self.in_features//2
        self.out_features = out_features
        self.scale = scale
        # self.theta = Parameter(torch.Tensor(self.out_features, self.in_dim))
        self.theta = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.reset_parameters(theta)

    def reset_parameters(self, theta = None) -> None:
        if theta is None:
            init.uniform_(self.theta, a=0, b=2*np.pi)
        else:
            assert theta.shape == (self.in_dim,self.out_features)
            self.theta = Parameter(theta) 
        self.real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
        cat_kernels_4_real = torch.cat(
            (self.real_kernel, -self.imag_kernel),
            dim=-1
        )
        cat_kernels_4_imag = torch.cat(
            (self.imag_kernel, self.real_kernel),
            dim=-1
        )
        cat_kernels_4_complex = torch.cat(
            (cat_kernels_4_real, cat_kernels_4_imag),
            dim=0
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]

        # output = F.linear(inputs, cat_kernels_4_complex)
        output = torch.matmul(inputs, cat_kernels_4_complex)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    def get_theta(self) -> torch.Tensor:
        return self.theta.detach().clone()
    
    def get_weights(self) -> torch.Tensor:
        with torch.no_grad():
            real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
            imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
            # cat_kernels_4_real = torch.cat(
            #     (real_kernel, -imag_kernel),
            #     dim=-1
            # )
            # cat_kernels_4_imag = torch.cat(
            #     (imag_kernel, real_kernel),
            #     dim=-1
            # )
            # cat_kernels_4_complex = torch.cat(
            #     (cat_kernels_4_real, cat_kernels_4_imag),
            #     dim=0
            # )  # This block matrix represents the conjugate transpose of the original:
            # # [ W_R, -W_I; W_I, W_R]
            beam_weights = real_kernel + 1j*imag_kernel
        return beam_weights



class DFT_Codebook_Layer(Module):
    def __init__(self, n_antenna, azimuths):
        super(DFT_Codebook_Layer, self).__init__()
        self.n_antenna = n_antenna
        self.n_beam = len(azimuths)
        dft_codebook = beam_utils.DFT_beam_blockmatrix(n_antenna = n_antenna, azimuths = azimuths)
        self.codebook_blockmatrix = torch.from_numpy(dft_codebook).float()
        self.codebook_blockmatrix.requires_grad = False
        self.codebook = beam_utils.DFT_beam(n_antenna = n_antenna, azimuths = azimuths).T
        
    def forward(self, x):
        bf_signal = torch.matmul(x, self.codebook_blockmatrix)
        return bf_signal
    
    def get_weights(self, x):
        return self.codebook
    

class ComputePower(Module):
    def __init__(self, in_shape):
        super(ComputePower, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[:,:self.len_real]
        imag_part = x[:,self.len_real:]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        return abs_values
    
class ComputePower_DoubleBatch(Module):
    def __init__(self, in_shape):
        super(ComputePower_DoubleBatch, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[...,:self.len_real]
        imag_part = x[...,self.len_real:]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        return abs_values

class Complex_Frobenius_Norm(Module):
    def __init__(self, in_shape):
        super(Complex_Frobenius_Norm, self).__init__()
        self.n_r = in_shape[0]
        self.n_c = in_shape[1]
        self.n_r_real = self.n_r//2
        self.n_c_real = self.n_c//2

    def forward(self, x): # x is b_size x n_r x n_c
        real_part = x[:,:self.n_r_real,:self.n_c_real]
        imag_part = x[:,self.n_r_real:,:self.n_c_real]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        abs_values = abs_values.sum(1).sum(1)
        fb_norm = torch.sqrt(abs_values)
        return fb_norm
        
class PowerPooling(Module):
    def __init__(self, in_shape):
        super(PowerPooling, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[:,:self.len_real]
        imag_part = x[:,self.len_real:]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        max_pooling = torch.max(abs_values, dim=-1)[0]
        max_pooling = torch.unsqueeze(max_pooling,dim=-1)
        return max_pooling

class Beam_Classifier(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
        super(Beam_Classifier, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
        else:            
            self.complex_codebook = complex_codebook # n_beams x n_antenna
            cb_blockmatrix = beam_utils.codebook_blockmatrix(self.complex_codebook.T)
            self.codebook = torch.from_numpy(cb_blockmatrix).float()
            self.codebook.requires_grad = False
            
        self.compute_power = ComputePower(2*n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        self.softmax = nn.Softmax()
    def forward(self, x):
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        noise_vec = torch.normal(0,1, size=bf_signal.size())*torch.sqrt(torch.tensor([self.noise_power/2]))/torch.tensor([self.norm_factor])
        bf_signal = bf_signal + noise_vec
        bf_power = self.compute_power(bf_signal)
        out = self.relu(bf_power)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().numpy()
        else:
            # return DFT_codebook(nseg=self.n_wide_beam,n_antenna=self.n_antenna).T
            return self.complex_codebook

def BF_loss(theta,h):
    num_antenna = theta.shape[1]
    phase = theta.unsqueeze(dim=1).transpose(1,2)
    bf_weights_real = torch.cos(phase)/torch.tensor(np.sqrt(num_antenna))
    bf_weights_imag = torch.sin(phase)/torch.tensor(np.sqrt(num_antenna))
    cat_kernels_4_real = torch.cat(
        (bf_weights_real, -bf_weights_imag),
        dim=-1
    )
    cat_kernels_4_imag = torch.cat(
        (bf_weights_imag, bf_weights_real),
        dim=-1
    )
    cat_kernels_4_complex = torch.cat(
        (cat_kernels_4_real, cat_kernels_4_imag),
        dim=-2
    )  # This block matrix represents the conjugate transpose of the original:
    # [ W_R, -W_I; W_I, W_R]    
    bf_signal = torch.matmul(h.unsqueeze(dim=1), cat_kernels_4_complex).squeeze()
    bf_signal_real = bf_signal[:,0]
    bf_signal_imag = bf_signal[:,1]
    sq_real = torch.pow(bf_signal_real,2)
    sq_imag = torch.pow(bf_signal_imag,2)
    bf_power = sq_real + sq_imag
    return -bf_power.mean()

def BF_gain_loss(theta,h,normalize=True):
    # h is n_batch x 2*num_antenna
    # theta is n_batch x num_antenna  
    num_antenna = torch.tensor([theta.shape[-1]]).to(h.device)
    phase = theta.unsqueeze(dim=1).transpose(1,2)
    bf_weights_real = torch.cos(phase)/torch.sqrt(num_antenna)
    bf_weights_imag = torch.sin(phase)/torch.sqrt(num_antenna)
    cat_kernels_4_real = torch.cat(
        (bf_weights_real, -bf_weights_imag),
        dim=-1
    )
    cat_kernels_4_imag = torch.cat(
        (bf_weights_imag, bf_weights_real),
        dim=-1
    )
    cat_kernels_4_complex = torch.cat(
        (cat_kernels_4_real, cat_kernels_4_imag),
        dim=-2
    )  # This block matrix represents the conjugate transpose of the original:
    # [ W_R, -W_I; W_I, W_R]    
    bf_signal = torch.matmul(h.unsqueeze(dim=1), cat_kernels_4_complex).squeeze()
    # print(h.shape, cat_kernels_4_complex.shape, bf_signal.shape)
    bf_signal_real = bf_signal[:,0]
    bf_signal_imag = bf_signal[:,1]
    sq_real = torch.pow(bf_signal_real,2)
    sq_imag = torch.pow(bf_signal_imag,2)
    bf_power = sq_real + sq_imag
    if normalize:
        norm_factor = torch.pow(h[:,:num_antenna],2) + torch.pow(h[:,num_antenna:],2) 
        norm_factor = norm_factor.sum(dim=-1)
        # print(bf_power.shape,norm_factor.shape)
        bf_power = bf_power/norm_factor      
    return -bf_power.mean()

def normalized_BF_loss(theta,h):
    num_antenna = theta.shape[1]
    phase = theta.unsqueeze(dim=1).transpose(1,2)
    bf_weights_real = torch.cos(phase)/torch.tensor(np.sqrt(num_antenna))
    bf_weights_imag = torch.sin(phase)/torch.tensor(np.sqrt(num_antenna))
    cat_kernels_4_real = torch.cat(
        (bf_weights_real, -bf_weights_imag),
        dim=-1
    )
    cat_kernels_4_imag = torch.cat(
        (bf_weights_imag, bf_weights_real),
        dim=-1
    )
    cat_kernels_4_complex = torch.cat(
        (cat_kernels_4_real, cat_kernels_4_imag),
        dim=-2
    )  # This block matrix represents the conjugate transpose of the original:
    # [ W_R, -W_I; W_I, W_R]    
    bf_signal = torch.matmul(h.unsqueeze(dim=1), cat_kernels_4_complex).squeeze()
    bf_signal_real = bf_signal[:,0]
    bf_signal_imag = bf_signal[:,1]
    sq_real = torch.pow(bf_signal_real,2)
    sq_imag = torch.pow(bf_signal_imag,2)
    bf_power = sq_real + sq_imag
    # normalized_gain = bf_power/egc_gain
    norm_factor = torch.pow(h[:,:num_antenna],2) + torch.pow(h[:,num_antenna:],2) 
    norm_factor = norm_factor.sum(dim=1)
    normalized_gain = bf_power/norm_factor
    return -normalized_gain.mean()

class Joint_Tx_Rx_Analog_Beamformer(Module):
    def __init__(self, num_antenna_Tx: int, num_antenna_Rx: int, num_beam_Tx: int, num_beam_Rx: int, 
                 theta_Tx = None, theta_Rx = None) -> None:
        super(Joint_Tx_Rx_Analog_Beamformer, self).__init__()
        self.num_antenna_Tx = num_antenna_Tx
        self.num_antenna_Rx = num_antenna_Rx
        self.num_beam_Tx = num_beam_Tx
        self.num_beam_Rx = num_beam_Rx
        # self.scale_Tx = torch.sqrt(torch.tensor([float(num_antenna_Tx)]))
        # self.scale_Rx = torch.sqrt(torch.tensor([float(num_antenna_Rx)]))
        self.register_buffer('scale_Tx',torch.sqrt(torch.tensor([num_antenna_Tx]).float()))
        self.register_buffer('scale_Rx',torch.sqrt(torch.tensor([num_antenna_Rx]).float()))
        self.theta_Tx = Parameter(torch.Tensor(self.num_antenna_Tx, self.num_beam_Tx)) 
        self.theta_Rx = Parameter(torch.Tensor(self.num_antenna_Rx, self.num_beam_Rx)) 
        self.reset_parameters(theta_Tx)
        self.reset_parameters(theta_Rx)

    def reset_parameters(self, theta_Tx: Tensor=None, theta_Rx: Tensor=None) -> None:
        if theta_Tx is None:
            init.uniform_(self.theta_Tx, a=0, b=2*np.pi)
        else:
            assert theta_Tx.shape == (self.num_antenna_Tx,self.num_beam_Tx)
            self.theta_Tx = Parameter(theta_Tx) 
        if theta_Rx is None:
            init.uniform_(self.theta_Rx, a=0, b=2*np.pi)
        else:
            assert theta_Rx.shape == (self.num_antenna_Rx,self.num_beam_Rx)
            self.theta_Rx = Parameter(theta_Rx) 
        
        self.real_kernel_Tx = torch.cos(self.theta_Tx) / self.scale_Tx
        self.imag_kernel_Tx = torch.sin(self.theta_Tx) / self.scale_Tx  
        self.real_kernel_Rx = torch.cos(self.theta_Rx) / self.scale_Rx
        self.imag_kernel_Rx = torch.sin(self.theta_Rx) / self.scale_Rx 
        
    def forward(self, h: Tensor) -> Tensor:
        # h is n_batch x 2*num_antenna_Rx x num_antenna_Tx
        rx_kernel = self.rx_kernel() # 2*num_beam_Rx* x 2*num_antenna_Rx
        # z = torch.matmul(h, rx_kernel) # z = W^H*h: n_batch x 2*num_beam_Rx x num_antenna_Tx
        z = torch.matmul(rx_kernel.transpose(-2,-1),h) # z = W^H*h: n_batch x 2*num_beam_Rx x num_antenna_Tx
        z_real = z[:,:self.num_beam_Rx,:]
        z_imag = z[:,self.num_beam_Rx:,:]
        z_complex = torch.cat(
            (z_real, z_imag),
            dim=-1
        ) # n_batch x num_beam_Rx x 2*num_antenna_Tx
        tx_kernel = self.tx_kernel() # 2*num_antenna_Tx x 2*num_beam_Tx
        output = torch.matmul(z_complex, tx_kernel) # n_batch x num_beam_Rx x 2*num_beam_Tx
        return output
    
    def tx_kernel(self) -> Tensor:
        self.real_kernel_Tx = torch.cos(self.theta_Tx) / self.scale_Tx
        self.imag_kernel_Tx = torch.sin(self.theta_Tx) / self.scale_Tx   
        cat_kernels_4_real = torch.cat(
            (self.real_kernel_Tx, self.imag_kernel_Tx),
            dim=-1
        )
        cat_kernels_4_imag = torch.cat(
            (-self.imag_kernel_Tx, self.real_kernel_Tx),
            dim=-1
        )
        cat_kernels_4_complex = torch.cat(
            (cat_kernels_4_real, cat_kernels_4_imag),
            dim=0
        )  # This block matrix represents the original:
        # [ W_R, W_I; -W_I, W_R]
        return cat_kernels_4_complex

    def rx_kernel(self) -> Tensor:
        self.real_kernel_Rx = torch.cos(self.theta_Rx) / self.scale_Rx
        self.imag_kernel_Rx = torch.sin(self.theta_Rx) / self.scale_Rx    
        cat_kernels_4_real = torch.cat(
            (self.real_kernel_Rx, -self.imag_kernel_Rx),
            dim=-1
        )
        cat_kernels_4_imag = torch.cat(
            (self.imag_kernel_Rx, self.real_kernel_Rx),
            dim=-1
        )
        cat_kernels_4_complex = torch.cat(
            (cat_kernels_4_real, cat_kernels_4_imag),
            dim=0
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]
        return cat_kernels_4_complex
            
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    def get_theta_Tx(self) -> torch.Tensor:
        return self.theta_Tx.detach().clone()

    def get_theta_Rx(self) -> torch.Tensor:
        return self.theta_Rx.detach().clone()
    
    def get_weights_Tx(self) -> torch.Tensor:
        with torch.no_grad():
            real_kernel = torch.cos(self.theta_Tx) / self.scale_Tx
            imag_kernel = torch.sin(self.theta_Tx) / self.scale_Tx   
            beam_weights = real_kernel + 1j*imag_kernel
        return beam_weights

    def get_weights_Rx(self) -> torch.Tensor:
        with torch.no_grad():
            real_kernel = torch.cos(self.theta_Rx) / self.scale_Rx
            imag_kernel = torch.sin(self.theta_Rx) / self.scale_Rx   
            beam_weights = real_kernel + 1j*imag_kernel
        return beam_weights
    
def joint_BF_gain_loss(tx_beam,rx_beam,h,normalize=False):
    # h is n_batch x 2*num_antenna_Rx x num_antenna_Tx
    # tx_beam is n_batch x num_antenna_Tx
    # rx_beam is n_batch x num_antenna_Rx
    num_antenna_Tx = torch.tensor([tx_beam.shape[-1]])
    scale_Tx = torch.sqrt(num_antenna_Tx.float()).to(h.device)
    num_antenna_Rx = torch.tensor([rx_beam.shape[-1]])
    scale_Rx = torch.sqrt(num_antenna_Rx.float()).to(h.device)
    tx_beam = tx_beam.unsqueeze(dim=-1) # n_batch x num_antenna_Tx x 1
    rx_beam = rx_beam.unsqueeze(dim=-1) # n_batch x num_antenna_Rx x 1
    real_kernel_Tx = torch.cos(tx_beam)/scale_Tx
    imag_kernel_Tx = torch.sin(tx_beam)/scale_Tx
    cat_kernels_4_real_tx = torch.cat(
        (real_kernel_Tx, imag_kernel_Tx),
        dim=-1
    )
    cat_kernels_4_imag_tx = torch.cat(
        (-imag_kernel_Tx, real_kernel_Tx),
        dim=-1
    )
    cat_kernels_4_complex_tx = torch.cat(
        (cat_kernels_4_real_tx, cat_kernels_4_imag_tx),
        dim=-2
    )  # This block matrix represents the original:
    # [ W_R, W_I; -W_I, W_R]   n_batch x 2*num_antenna_Tx x 2
    real_kernel_Rx = torch.cos(rx_beam)/scale_Rx
    imag_kernel_Rx = torch.sin(rx_beam)/scale_Rx  
    cat_kernels_4_real_rx = torch.cat(
        (real_kernel_Rx, -imag_kernel_Rx),
        dim=-1
    )
    cat_kernels_4_imag_rx = torch.cat(
        (imag_kernel_Rx, real_kernel_Rx),
        dim=-1
    )
    cat_kernels_4_complex_rx = torch.cat(
        (cat_kernels_4_real_rx, cat_kernels_4_imag_rx),
        dim=-2
    )  # This block matrix represents the conjugate transpose of the original:
    # [ W_R, -W_I; W_I, W_R]   n_batch x 2*num_antenna_Rx x 2
    z = torch.matmul(cat_kernels_4_complex_rx.transpose(-2,-1),h) # z = W^H*h: n_batch x 2 x num_antenna_Tx
    z_real = z[:,[0],:]
    z_imag = z[:,[1],:]
    z_complex = torch.cat(
        (z_real, z_imag),
        dim=-1
    ) # n_batch x 1 x 2*num_antenna_Tx
    y = torch.matmul(z_complex, cat_kernels_4_complex_tx).squeeze(dim=1) # n_batch x 1 x 2 -> n_batch x 2
    y_real = y[:,0]
    y_imag = y[:,1]
    y_real_sq = torch.pow(y_real,2)
    y_imag_sq = torch.pow(y_imag,2)
    bf_power = y_real_sq + y_imag_sq
    if normalize:
        norm_factor = torch.pow(h[:,:num_antenna_Rx,:],2) + torch.pow(h[:,num_antenna_Rx:,:],2) 
        # norm_factor = norm_factor.sum(dim=(1,2)).unsqueeze(dim=-1)
        norm_factor = norm_factor.sum(dim=(1,2))
        bf_power = bf_power/norm_factor        
    return -bf_power.mean()