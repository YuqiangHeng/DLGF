import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook,UPA_DFT_codebook
from DL_utils_MISO import BF_Autoencoder,BF_DFT_Autoencoder,DirectBF_CB,BF_gain_loss,spectral_efficiency_loss,fit_GF,fit_CB

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import DeepMIMO
import argparse
from os.path import exists

parser = argparse.ArgumentParser()
parser.add_argument('--num_probing_beam', type=int, default=32)
parser.add_argument('--measurement_gain', type=float, default=32)
parser.add_argument('--Tx_power_dBm', type=int, default=20)
parser.add_argument('--nepoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=800)
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--scenario',type=str,default='Boston5G_28')
parser.add_argument('--activated_BS',type=int,default=2)
parser.add_argument('--h_NMSE_dB',type=int,default=-20)
parser.add_argument('--dataset_split_seed',type=int,default=7)
parser.add_argument('--IA_threshold',type=int,default=-5)
parser.add_argument('--use_specific_Tx_power', dest='use_default_Tx_power', default=True, action='store_false')
parser.add_argument('--array_type',type=str,default='ULA')
parser.add_argument('--probing_codebook',type=str,default='learned',help='learned, DFT')
parser.add_argument('--BW',type=float,default=50,help='MHz')
parser.add_argument('--noise_PSD_dB',type=float,default=-161,help='dBm/Hz')
parser.add_argument('--loss_fn',type=str,default='BF_loss',help='SPE_loss, BF_loss, CE, BCE')
parser.add_argument('--mode',type=str,default='GF',help='GF, CB, Li19_CB')
parser.add_argument('--oversample_factor',type=float,default=4)

args = parser.parse_args()

model_savefname_prefix = './Saved_Models/'
train_hist_savefname_prefix = './Train_Hist/'

if args.use_default_Tx_power:
    if args.scenario == 'O1_28':
        tx_power_dBm = 20
    elif args.scenario == 'O1_28B':
        tx_power_dBm = 35    
    elif args.scenario == 'I3_60':
        if args.activated_BS==1:
            tx_power_dBm = 15
        elif args.activated_BS==2:
            tx_power_dBm = 20
        else:
            raise Exception("Unsupported BS activation!") 
    elif args.scenario == 'Boston5G_28':
        tx_power_dBm = 40
    else:
        raise Exception("Unsupported Ray-Tracing Scenario!")
else:
    tx_power_dBm = args.Tx_power_dBm

# n_probing_beam = args.num_probing_beam
noise_power_dBm = args.noise_PSD_dB + 10*np.log10(args.BW*1e6) # dBm
measurement_gain = args.measurement_gain
measurement_gain_dB = 10*np.log10(measurement_gain)
nepoch = args.nepoch
batch_size = args.batch_size  
IA_threshold = args.IA_threshold
measurement_noise_power = 10**((noise_power_dBm-tx_power_dBm)/10)/measurement_gain
h_NMSE = 10**(args.h_NMSE_dB/10)

gpu_name = "cuda:{}".format(args.gpu)
device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
print(device)

if args.scenario == 'O1_28':
    BS_arr = np.array([3])
    ue_row_first = 800
    ue_row_last = 1200
elif args.scenario == 'O1_28B':
    BS_arr = np.array([3])
    ue_row_first = 1
    ue_row_last = 2751    
elif args.scenario == 'I3_60':
    BS_arr = np.array([args.activated_BS])
    # LOS grid: 1-551; NLOS grid: 552-1159
    ue_row_first = 1
    ue_row_last = 551
elif args.scenario == 'Boston5G_28':
    BS_arr = np.array([1])
    ue_row_first = 1
    ue_row_last = 1622
else:
    raise Exception("Unsupported Ray-Tracing Scenario!")

parameters = DeepMIMO.default_params()
parameters['dataset_folder'] = 'D:/Github Repositories/DeepMIMO-codes/DeepMIMOv2/Raytracing_scenarios'
parameters['scenario'] = args.scenario
parameters['num_paths'] = 15
parameters['active_BS'] = BS_arr
parameters['user_row_first'] = ue_row_first
parameters['user_row_last'] = ue_row_last
parameters['row_subsampling'] = 1
parameters['user_subsampling'] = 1

parameters['ue_antenna']['shape'] = np.array([1, 1, 1])

if args.array_type == 'UPA':
    BS_array_shape = np.array([1, 8, 8])
    parameters['bs_antenna']['shape'] = BS_array_shape
    num_antenna_az = BS_array_shape[1]
    num_antenna_el = BS_array_shape[2]
elif args.array_type == 'ULA':
    BS_array_shape = np.array([64, 1, 1])
    parameters['bs_antenna']['shape'] = BS_array_shape
else:
    raise Exception("Unsupported Antenna Array Type!")
num_antenna = np.prod(parameters['bs_antenna']['shape'])


parameters['bs_antenna']['spacing'] = 0.5
parameters['bs_antenna']['radiation_pattern'] = 'isotropic'  

# parameters['ue_antenna']['spacing'] = 0.5
parameters['ue_antenna']['radiation_pattern'] = 'isotropic'


parameters['enable_BS2BS'] = 0

parameters['OFDM_channels'] = 1 # Frequency (OFDM) or time domain channels
parameters['OFDM']['subcarriers'] = 512
parameters['OFDM']['subcarriers_limit'] = 1
parameters['OFDM']['subcarriers_sampling'] = 1
parameters['OFDM']['bandwidth'] = args.BW/1e3 # GHz
parameters['OFDM']['RX_filter'] = 0

dataset_savefname = './Data/{}_{}x{}x{}.npy'.format(args.scenario,
                                                BS_array_shape[0],
                                                BS_array_shape[1],
                                                BS_array_shape[2])

if exists(dataset_savefname):
    print('Loading dataset...')
    h = np.load(dataset_savefname,allow_pickle=True)
else:
    print('Generating dataset...')
    dataset = DeepMIMO.generate_data(parameters)
    h = dataset[0]['user']['channel'].squeeze(axis=-1)
    valid_ue_idc = np.array([ue_i for (ue_i,ue_h) in enumerate(h) if not (ue_h==0).all()])
    print('Keep {} out of {} UEs that have valid paths.'.format(len(valid_ue_idc),h.shape[0]))
    h = h[valid_ue_idc]
    np.save(dataset_savefname,h,allow_pickle=True)

if args.IA_threshold > -np.inf:
    EGT_gain_path = './Data/EGT_gain_{}_BS_{}_BS_array_{}x{}x{}.npy'.format(parameters['scenario'],
                                                                               parameters['active_BS'][0],
                                                                               BS_array_shape[0],
                                                                               BS_array_shape[1],
                                                                               BS_array_shape[2])
    if exists(EGT_gain_path):
        EGT_gain = np.load(EGT_gain_path,allow_pickle=True)
    else:
        # MRT_gain = np.linalg.norm(h.squeeze(),axis=-1)**2
        EGT_gain = np.power(np.sum(abs(h.squeeze()),axis=1),2)/num_antenna
        np.save(EGT_gain_path,EGT_gain,allow_pickle=True)
    EGT_snr = tx_power_dBm+10*np.log10(EGT_gain)-noise_power_dBm
    reachable_ue = EGT_snr>args.IA_threshold   
    print('Keep {} out of {} UEs that are above the IA SNR threshold ({} dB).'.format(reachable_ue.sum(),h.shape[0],args.IA_threshold))    
    h = h[reachable_ue]

norm_factor = np.max(abs(h))
h_scaled = (h.T/norm_factor).T
    
# Training and testing data:
# --------------------------
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4,random_state=args.dataset_split_seed)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5,random_state=args.dataset_split_seed)
# train_idc, val_idc, test_idc = np.arange(1),np.arange(1),np.arange(1)

x_train,x_val,x_test = h_scaled[train_idc],h_scaled[val_idc],h_scaled[test_idc]

np.random.seed(args.dataset_split_seed)
for i,h_iter in enumerate(x_train):
    h_noise_scale = np.sqrt(abs(h_iter)**2*h_NMSE/2)
    h_train_noise = (np.random.normal(size=h_iter.shape)+1j*np.random.normal(size=h_iter.shape))*h_noise_scale
    x_train[i] = h_iter+h_train_noise

torch_x_train = torch.from_numpy(x_train).to(device)
torch_x_val = torch.from_numpy(x_val).to(device)
torch_x_test = torch.from_numpy(x_test).to(device)
num_beams  = None
if args.mode in ['CB','Li19_CB']:
    if args.array_type == 'UPA':
        num_beams = int(num_antenna_az*args.oversample_factor)*int(num_antenna_el*args.oversample_factor)
        DFT_codebook = UPA_DFT_codebook(n_azimuth=int(num_antenna_az*args.oversample_factor),
                                        n_elevation=int(num_antenna_el*args.oversample_factor),
                                        n_antenna_azimuth=num_antenna_az,n_antenna_elevation=num_antenna_el,spacing=0.5).T
    else:
        num_beams = int(num_antenna*args.oversample_factor)
        DFT_codebook = ULA_DFT_codebook(nseg=num_beams,n_antenna=num_antenna).T

    DFT_beam_target_path = './Data/DFT_beam_target_{}_BS_{}_BS_array_{}x{}x{}_oversample_{}.npy'.format(parameters['scenario'],parameters['active_BS'][0],
                                                                                   BS_array_shape[0],
                                                                                   BS_array_shape[1],
                                                                                   BS_array_shape[2],
                                                                                   args.oversample_factor)
    if exists(DFT_beam_target_path):
        DFT_beam_target = np.load(DFT_beam_target_path,allow_pickle=True)
    else:
        dft_bf_gain = abs(h @ DFT_codebook).squeeze()**2
        DFT_beam_target = np.argmax(dft_bf_gain,1) 
        np.save(DFT_beam_target_path,DFT_beam_target,allow_pickle=True)   
    if args.mode == 'Li19_CB' and args.loss_fn == 'BCE':
        DFT_beam_target_onehot = np.zeros((DFT_beam_target.shape[0],num_beams),dtype=np.float32)
        DFT_beam_target_onehot[np.arange(DFT_beam_target.shape[0]),DFT_beam_target] = 1
        DFT_beam_target = DFT_beam_target_onehot
    torch_y_train = torch.from_numpy(DFT_beam_target[train_idc]).to(device)
    torch_y_val = torch.from_numpy(DFT_beam_target[val_idc]).to(device)
    torch_y_test = torch.from_numpy(DFT_beam_target[test_idc]).to(device)
    train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train) 
    val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
    test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)
else:
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(torch_x_train)
    val = torch.utils.data.TensorDataset(torch_x_val)
    test = torch.utils.data.TensorDataset(torch_x_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

# for n_probing_beam in [2,4,8]:
# for n_probing_beam in [12,16,20]:
# for n_probing_beam in [24,28,32]:
for n_probing_beam in [36,48,64]:
# for n_probing_beam in [2,4,8,12,16,20,24,28,32,48]:
# for n_probing_beam in [60]:
    print('num {} probing beams = {}.'.format(args.probing_codebook,n_probing_beam))
    if args.mode == 'GF':
        mode_savename = args.mode
    else:
        mode_savename = args.mode+'_'+str(num_beams)
    if args.probing_codebook == 'learned':
        autoencoder = BF_Autoencoder(num_antenna =num_antenna,num_probing_beam = n_probing_beam,
                                    noise_power=measurement_noise_power, norm_factor = norm_factor,
                                    mode = args.mode, num_beam = num_beams).to(device)
    else:
        autoencoder = BF_DFT_Autoencoder(num_antenna =num_antenna,num_probing_beam = n_probing_beam,
                                    noise_power=measurement_noise_power, norm_factor = norm_factor,
                                    mode = args.mode, num_beam = num_beams).to(device)
    if args.mode == 'Li19_CB':
        # autoencoder = DirectBF_CB(n_antenna =num_antenna,n_wide_beam = n_probing_beam, n_narrow_beam = num_beams,
        #                             noise_power=measurement_noise_power, norm_factor = norm_factor).to(device)
        autoencoder = DirectBF_CB(num_antenna =num_antenna,num_probing_beam = n_probing_beam,
                                    noise_power=measurement_noise_power, norm_factor = norm_factor,
                                    loss_fn = args.loss_fn, num_beam = num_beams).to(device)        
    model_setup_params = ("MISO_{}_BS_{}_{}_"+
                    "{}_{}_{}_Nprobe_{}_"+
                    "{}_antenna").format(parameters['scenario'],parameters['active_BS'][0],args.array_type,
                                            mode_savename,args.probing_codebook,args.loss_fn,
                                            n_probing_beam,num_antenna)        
    model_savefname = model_savefname_prefix+model_setup_params+".pt"
    autoencoder_opt = optim.Adam(autoencoder.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=True)

    if args.mode == 'GF':   
        if args.loss_fn == 'SPE_loss':
            loss_fn = spectral_efficiency_loss(scale=norm_factor,noise_power_dBm=noise_power_dBm,Tx_power_dBm=tx_power_dBm) 
            train_loss_hist, val_loss_hist = fit_GF(autoencoder, train_loader, val_loader, autoencoder_opt, loss_fn, nepoch, train_hist_savefname_prefix+model_setup_params)  
        elif args.loss_fn == 'BF_loss':
            train_loss_hist, val_loss_hist = fit_GF(autoencoder, train_loader, val_loader, autoencoder_opt, BF_gain_loss, nepoch, train_hist_savefname_prefix+model_setup_params)   
    elif args.mode == 'Li19_CB':
        if args.loss_fn == 'CE':
            train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = fit_CB(autoencoder, train_loader, val_loader, 
                                                                                autoencoder_opt, nn.CrossEntropyLoss(), nepoch, 
                                                                                train_hist_savefname_prefix+model_setup_params,loss_fn_name=args.loss_fn) 
        else:
            train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = fit_CB(autoencoder, train_loader, val_loader, 
                                                                                autoencoder_opt, nn.BCELoss(), nepoch, 
                                                                                train_hist_savefname_prefix+model_setup_params,loss_fn_name=args.loss_fn)                    
    else:
        train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = fit_CB(autoencoder, train_loader, val_loader, 
                                                                            autoencoder_opt, nn.CrossEntropyLoss(), nepoch, 
                                                                            train_hist_savefname_prefix+model_setup_params,loss_fn_name='CE')
    autoencoder = autoencoder.cpu()
    torch.save(autoencoder.state_dict(),model_savefname)

    plt.figure()
    plt.plot(train_loss_hist,label='training')
    plt.plot(val_loss_hist,label='validation')
    plt.legend()
    plt.title('loss hist: {} probing beams'.format(n_probing_beam))
    plt.savefig(train_hist_savefname_prefix+model_setup_params+"_loss.png")

    if args.mode == 'CB':
        plt.figure()
        plt.plot(train_acc_hist,label='training')
        plt.plot(val_acc_hist,label='validation')
        plt.legend()
        plt.title('Acc hist: {} probing beams'.format(n_probing_beam))
        plt.savefig(train_hist_savefname_prefix+model_setup_params+"_acc.png")
