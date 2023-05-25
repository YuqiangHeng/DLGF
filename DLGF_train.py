import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import DeepMIMO
import argparse
import time
import pickle
from os.path import exists
from DL_utils import Joint_BF_Autoencoder,combined_BF_IA_loss,spectral_efficiency_loss,BF_loss,fit,fit_SPE

# torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--num_probing_beam', type=int, default=64)
parser.add_argument('--measurement_gain', type=float, default=1)
parser.add_argument('--Tx_power_dBm', type=int, default=20)
parser.add_argument('--nepoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=800)
parser.add_argument('--gamma',type=float, default=0.3)
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--scenario',type=str,default='Boston5G_28')
parser.add_argument('--activated_BS',type=int,default=2)
parser.add_argument('--h_NMSE_dB',type=int,default=-np.inf)
parser.add_argument('--dataset_split_seed',type=int,default=7)
parser.add_argument('--IA_threshold',type=int,default=-np.inf)
parser.add_argument('--num_feedback',default=None)
parser.add_argument('--UE_rotation', dest='random_UE_rotation', default=False, action='store_true')
parser.add_argument('--use_specific_Tx_power', dest='use_default_Tx_power', default=True, action='store_false')
parser.add_argument('--array_type',type=str,default='ULA')
parser.add_argument('--feedback_mode',type=str,default='diagonal')
parser.add_argument('--beam_synthesizer',type=str,default='MLP')
parser.add_argument('--learned_probing',type=str,default='TxRx')
parser.add_argument('--BW',type=float,default=50,help='MHz')
parser.add_argument('--noise_PSD_dB',type=float,default=-143,help='dBm/Hz')
parser.add_argument('--loss_fn',type=str,default='BF_IA_loss',help='BF_IA_loss, SPE_loss, BF_loss')

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

n_probing_beam = args.num_probing_beam
noise_power_dBm = args.noise_PSD_dB + 10*np.log10(args.BW*1e6) # dBm
measurement_gain = args.measurement_gain
measurement_gain_dB = 10*np.log10(measurement_gain)
nepoch = args.nepoch
batch_size = args.batch_size  
gamma = args.gamma # trade-off between BF loss and misdetection loss
IA_threshold = args.IA_threshold
measurement_noise_power = 10**((noise_power_dBm-tx_power_dBm)/10)/measurement_gain

if args.num_feedback is None:
    num_feedback = args.num_feedback
else:
    num_feedback = int(args.num_feedback)
    
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

if args.array_type == 'UPA':
    parameters['bs_antenna']['shape'] = np.array([1, 8, 8])
    parameters['ue_antenna']['shape'] = np.array([1, 4, 4])
elif args.array_type == 'ULA':
    parameters['bs_antenna']['shape'] = np.array([64, 1, 1])
    parameters['ue_antenna']['shape'] = np.array([32, 1, 1])
else:
    raise Exception("Unsupported Antenna Array Type!")

parameters['bs_antenna']['spacing'] = 0.5
parameters['bs_antenna']['radiation_pattern'] = 'isotropic'  

parameters['ue_antenna']['spacing'] = 0.5
parameters['ue_antenna']['radiation_pattern'] = 'isotropic'

if args.random_UE_rotation and args.array_type == 'UPA':
    parameters['ue_antenna']['rotation'] = np.array([[-180,180],[-90,90],[-90,90]])
if args.random_UE_rotation and args.array_type == 'ULA':
    parameters['ue_antenna']['rotation'] = np.array([[-180,180],[0,0],[0,0]])


parameters['enable_BS2BS'] = 0

parameters['OFDM_channels'] = 1 # Frequency (OFDM) or time domain channels
parameters['OFDM']['subcarriers'] = 512
parameters['OFDM']['subcarriers_limit'] = 1
parameters['OFDM']['subcarriers_sampling'] = 1
parameters['OFDM']['bandwidth'] = args.BW/1e3 # GHz
parameters['OFDM']['RX_filter'] = 0


dataset = DeepMIMO.generate_data(parameters)
h = dataset[0]['user']['channel'].squeeze(axis=-1)
valid_ue_idc = np.array([ue_i for (ue_i,ue_h) in enumerate(h) if not (ue_h==0).all()])
print('Keep {} out of {} UEs that have valid paths.'.format(len(valid_ue_idc),h.shape[0]))
h = h[valid_ue_idc]

if args.IA_threshold > -np.inf:
    eigen_bf_gain_path = 'eigen_bf_gain_{}_BS_{}_BS_array_{}x{}x{}_UE_array_{}x{}x{}.npy'.format(parameters['scenario'],parameters['active_BS'][0],
                                                                                            parameters['bs_antenna'][0]['shape'][0],parameters['bs_antenna'][0]['shape'][1],parameters['bs_antenna'][0]['shape'][2],
                                                                                            parameters['ue_antenna']['shape'][0],parameters['ue_antenna']['shape'][1],parameters['ue_antenna']['shape'][2])
    # if exists(eigen_bf_gain_path):
    #     eigen_bf_gain = np.load(eigen_bf_gain_path,allow_pickle=True)
    # else:
    eigen_bf_gain = np.linalg.eigvalsh(np.transpose(h.conj(),axes=(0,2,1)) @ h)[:,-1]
        # np.save(eigen_bf_gain_path,eigen_bf_gain,allow_pickle=True)
    eigen_snr = tx_power_dBm+10*np.log10(eigen_bf_gain)-noise_power_dBm
    reachable_ue = eigen_snr>args.IA_threshold   
    print('Removed {} out of {} UEs that are below the IA SNR threshold ({} dB).'.format((~reachable_ue).sum(),h.shape[0],args.IA_threshold))    
    h = h[reachable_ue]

norm_factor = np.max(abs(h))
h_scaled = (h.T/norm_factor).T

# if args.random_UE_rotation:
#     ue_rotation = parameters['ue_antenna']['rotation']
#     ue_rotation = ue_rotation[reachable_ue]/np.array([360,180,180])+0.5 # map to 0-1
#     ue_rotation = ue_rotation.astype(np.single)

# Training and testing data:
# --------------------------
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4,random_state=args.dataset_split_seed)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5,random_state=args.dataset_split_seed)

x_train,x_val,x_test = h_scaled[train_idc],h_scaled[val_idc],h_scaled[test_idc]

torch_x_train = torch.from_numpy(x_train).to(device)
torch_x_val = torch.from_numpy(x_val).to(device)
torch_x_test = torch.from_numpy(x_test)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_x_train)
val = torch.utils.data.TensorDataset(torch_x_val)
test = torch.utils.data.TensorDataset(torch_x_test)


# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = len(val), shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = len(test), shuffle = False)
    
num_antenna_Tx = np.prod(parameters['bs_antenna'][0]['shape'])
num_antenna_Rx = np.prod(parameters['ue_antenna']['shape'])

print('num probing beams = {}.'.format(n_probing_beam))

autoencoder = Joint_BF_Autoencoder(num_antenna_Tx = num_antenna_Tx, num_antenna_Rx = num_antenna_Rx, 
                             num_probing_beam_Tx = n_probing_beam, num_probing_beam_Rx = n_probing_beam, 
                             noise_power=measurement_noise_power, norm_factor=norm_factor,
                             feedback=args.feedback_mode,num_feedback = args.num_feedback,
                             learned_probing = args.learned_probing, beam_synthesizer = args.beam_synthesizer).to(device)

if args.loss_fn != 'BF_IA_loss':
    gamma = None

model_setup_params = ("{}_BS_{}_"+
                   "{}x{}_{}_"+
                   "UE_rot_{}_"+
                   "{}_{}_probe_"+
                   "{}_FB_{}_"+
                   "{}_gamma_{}_"+
                   "train_noise_{}_dBm_"+
                   "meas_gain_{}_"+
                   "IA_thresh_{}_"+
                   "h_NMSE_{}").format(parameters['scenario'], parameters['active_BS'][0],
                                             num_antenna_Tx, num_antenna_Rx, args.array_type,
                                             args.random_UE_rotation,
                                             n_probing_beam, args.learned_probing,                                        
                                             args.feedback_mode, num_feedback,
                                             args.loss_fn, gamma,
                                             noise_power_dBm,
                                             measurement_gain,
                                             args.IA_threshold,
                                             args.h_NMSE_dB)
model_savefname = model_savefname_prefix+model_setup_params+".pt"

autoencoder_opt = optim.Adam(autoencoder.parameters(),lr=0.0001, betas=(0.9,0.999), amsgrad=True)
t_start = time.time()
if args.loss_fn == 'BF_IA_loss':
    loss_fn = combined_BF_IA_loss(scale=norm_factor,gamma=gamma,snr_threshold=args.IA_threshold,noise_power_dBm=noise_power_dBm,Tx_power_dBm=tx_power_dBm)  
    train_hist = fit(autoencoder, train_loader, val_loader, autoencoder_opt, loss_fn, nepoch, model_savefname = train_hist_savefname_prefix+model_setup_params, h_NMSE_dB = args.h_NMSE_dB)  
elif args.loss_fn == 'SPE_loss':
    loss_fn = spectral_efficiency_loss(scale=norm_factor,noise_power_dBm=noise_power_dBm,Tx_power_dBm=tx_power_dBm) 
    train_hist = fit_SPE(autoencoder, train_loader, val_loader, autoencoder_opt, loss_fn, nepoch, model_savefname = train_hist_savefname_prefix+model_setup_params, device=device)  
elif args.loss_fn == 'BF_loss':
    loss_fn = BF_loss(noise_power_dBm=noise_power_dBm,Tx_power_dBm=tx_power_dBm)   
    train_hist = fit_SPE(autoencoder, train_loader, val_loader, autoencoder_opt, loss_fn, nepoch, model_savefname = train_hist_savefname_prefix+model_setup_params, device=device)   
else:
    raise Exception("Unsupported Loss Function!")

t_end = time.time()
print('time to train {} epochs with batch size {}: {:.5f} min'.format(nepoch,batch_size,(t_end-t_start)/60))
autoencoder = autoencoder.cpu()
torch.save(autoencoder.state_dict(),model_savefname)

loss_hist_savefname = train_hist_savefname_prefix+model_setup_params+'.npy'

np.save(loss_hist_savefname,train_hist,allow_pickle=True)

# fig, axs = plt.subplots(2, 2, figsize=(10,8))
# axs[0, 0].plot(loss_hist['combined loss'][0], label='training')
# axs[0, 0].plot(loss_hist['combined loss'][1], label='validation')
# axs[0, 0].set_title('combined loss')
# axs[0, 0].set_xlabel('epoch')
# axs[0, 1].plot(loss_hist['BF loss'][0], label='training')
# axs[0, 1].plot(loss_hist['BF loss'][1], label='validation')
# axs[0, 1].set_title('BF loss')
# axs[0, 1].set_xlabel('epoch')
# axs[1, 0].plot(loss_hist['misdetection loss'][0], label='training')
# axs[1, 0].plot(loss_hist['misdetection loss'][1], label='validation')
# axs[1, 0].set_title('misdetection loss')
# axs[1, 0].set_xlabel('epoch')
# axs[1, 1].plot(loss_hist['misdetection rate'][0], label='training')
# axs[1, 1].plot(loss_hist['misdetection rate'][1], label='validation')
# axs[1, 1].set_title('misdetection rate')
# axs[1, 1].set_xlabel('epoch')
# fig.tight_layout()
# fig.savefig(train_hist_savefname_prefix+model_setup_params+'.pdf',dpi=300,format='pdf')