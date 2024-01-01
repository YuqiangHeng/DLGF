import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import DeepMIMO
import argparse
import time
from os.path import exists
from DL_utils import Joint_BF_Classifier,fit_CB
from beam_utils import UPA_DFT_codebook, unravel_index


torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--num_probing_beam', type=int, default=48, help='number of probing beams')
parser.add_argument('--measurement_gain', type=float, default=16.0, help='spreading gain of the probing measurements')
parser.add_argument('--Tx_power_dBm', type=int, default=20, help='Tx power in dBm')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=800, help='batch size')
parser.add_argument('--gamma',type=float, default=1.0, help='gamma is the weight of the normalized BF loss, 1-gamma is the weight of the IA loss')
parser.add_argument('--gpu',type=int,default=0, help='gpu id to train on')
parser.add_argument('--scenario',type=str,default='O1_28', help='DeepMIMO Ray-tracing scenario')
parser.add_argument('--activated_BS',type=int,default=2, help='Index of the activated BS in the DeepMIMO scenario')
parser.add_argument('--h_NMSE_dB',type=int,default=-np.inf, help='channel estimation error in dB, to generate noisy training data')
parser.add_argument('--dataset_split_seed',type=int,default=7, help='random seed for partitioning the dataset into training and testing')
parser.add_argument('--IA_threshold',type=int,default=-5, help='IA threshold in dB, UE need to achieve highe SNR than this threshold with one of the probing beams for initial access')
parser.add_argument('--num_feedback',default=None, help='number of probing beam measuements to feed back to the BS, None means all the measurements are fed back')
parser.add_argument('--UE_rotation', dest='random_UE_rotation', default=False, action='store_true', help='randomly rotate the UE antenna array')
parser.add_argument('--use_specific_Tx_power', dest='use_default_Tx_power', default=True, action='store_false', help='use specific Tx power instead of the default value')
parser.add_argument('--array_type',type=str,default='UPA', help='BS and UE array type, ULA or UPA')
parser.add_argument('--feedback_mode',type=str,default='diagonal', help='feedback mode for the probing measurements, consider measurements of all Tx/Rx beam pairs as a square matrix, feed back diagonal elements or full')
parser.add_argument('--beam_synthesizer',type=str,default='MLP', help='beam synthesizer architecture, MLP or CNN')
parser.add_argument('--learned_probing',type=str,default='TxRx', help='TxRx: learn Tx and Rx probing beams jointly; Tx: learn Tx probing beams only; Rx: learn Rx probing beams only')
parser.add_argument('--BW',type=float,default=50, help='bandwidth in MHz')
parser.add_argument('--noise_PSD_dB',type=float,default=-143, help='noise PSD in dBm/Hz')
args = parser.parse_args()

model_savefname_prefix = './Saved_Models/'
train_hist_savefname_prefix = './Train_Hist/'

# Default Tx power for each scenario
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
IA_threshold = args.IA_threshold
measurement_noise_power = 10**((noise_power_dBm-tx_power_dBm)/10)/measurement_gain

if args.num_feedback is None:
    num_feedback = args.num_feedback
else:
    num_feedback = int(args.num_feedback)
    
gpu_name = "cuda:{}".format(args.gpu)
device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
print(device)

# specify the UE grid depending on the scenario
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
    
# DeepMIMO parameters
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
valid_ue_idc = np.array([ue_i for (ue_i,ue_h) in enumerate(h) if not (ue_h==0).all()]) # remove UEs that have all zero paths
print('Removed {} out of {} UEs that have no valid paths.'.format(len(valid_ue_idc),h.shape[0]))
h = h[valid_ue_idc]

if args.IA_threshold > -np.inf:
    eigen_bf_gain_path = './Data/eigen_bf_gain_{}_BS_{}_BS_array_{}x{}x{}_UE_array_{}x{}x{}.npy'.format(parameters['scenario'],parameters['active_BS'][0],
                                                                                            parameters['bs_antenna'][0]['shape'][0],parameters['bs_antenna'][0]['shape'][1],parameters['bs_antenna'][0]['shape'][2],
                                                                                            parameters['ue_antenna']['shape'][0],parameters['ue_antenna']['shape'][1],parameters['ue_antenna']['shape'][2])
    if exists(eigen_bf_gain_path):
        eigen_bf_gain = np.load(eigen_bf_gain_path,allow_pickle=True)
    else:
        eigen_bf_gain = np.linalg.eigvalsh(np.transpose(h.conj(),axes=(0,2,1)) @ h)[:,-1] # compute the max eigenvalue of the channel covariance matrix, this is the uppero bound of the BF gain
        np.save(eigen_bf_gain_path,eigen_bf_gain,allow_pickle=True)
    eigen_snr = tx_power_dBm+10*np.log10(eigen_bf_gain)-noise_power_dBm # SNR with eigen beamforming
    reachable_ue = eigen_snr>args.IA_threshold   # remove UEs that are impossible to complete IA with eigen beamforming
    print('Removed {} out of {} UEs that are below the IA SNR threshold ({} dB).'.format((~reachable_ue).sum(),h.shape[0],args.IA_threshold))    
    h = h[reachable_ue]

# normalization of the channel matrices help with training
norm_factor = np.max(abs(h))
h_scaled = (h.T/norm_factor).T

# if args.random_UE_rotation:
#     ue_rotation = parameters['ue_antenna']['rotation']
#     ue_rotation = ue_rotation[reachable_ue]/np.array([360,180,180])+0.5 # map to 0-1
#     ue_rotation = ue_rotation.astype(np.single)

# --------------------------
# compute the best DFT beam pair for each channel
DFT_codebook_TX = UPA_DFT_codebook(n_azimuth=8*2,n_elevation=8*2,n_antenna_azimuth=8,n_antenna_elevation=8,spacing=0.5).T
DFT_codebook_RX = UPA_DFT_codebook(n_azimuth=4*2,n_elevation=4*2,n_antenna_azimuth=4,n_antenna_elevation=4,spacing=0.5).T

DFT_beam_target_path = './Data/DFT_beam_target_{}_BS_{}_BS_array_{}x{}x{}_UE_array_{}x{}x{}.npy'.format(parameters['scenario'],parameters['active_BS'][0],
                                                                                        parameters['bs_antenna'][0]['shape'][0],parameters['bs_antenna'][0]['shape'][1],parameters['bs_antenna'][0]['shape'][2],
                                                                                        parameters['ue_antenna']['shape'][0],parameters['ue_antenna']['shape'][1],parameters['ue_antenna']['shape'][2])
if  exists(DFT_beam_target_path):
    DFT_beam_target = np.load(DFT_beam_target_path,allow_pickle=True).item()
    best_rx_dft_beam_idx = DFT_beam_target['best_rx_dft_beam_idx']
    best_tx_dft_beam_idx = DFT_beam_target['best_tx_dft_beam_idx']
else:
    dft_bf_gain = abs(DFT_codebook_RX.conj().T @ h @ DFT_codebook_TX)**2
    best_rx_dft_beam_idx, best_tx_dft_beam_idx = np.unravel_index(dft_bf_gain.reshape(dft_bf_gain.shape[0],-1).argmax(1),dft_bf_gain[0].shape)
    DFT_beam_target = {}
    DFT_beam_target['best_rx_dft_beam_idx'] = best_rx_dft_beam_idx
    DFT_beam_target['best_tx_dft_beam_idx'] = best_tx_dft_beam_idx
    np.save(DFT_beam_target_path,DFT_beam_target,allow_pickle=True)

# Training and testing data:
# --------------------------
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4,random_state=args.dataset_split_seed)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5,random_state=args.dataset_split_seed)

h_train,h_val = h_scaled[train_idc],h_scaled[val_idc]
tx_label_train,tx_label_val = best_tx_dft_beam_idx[train_idc],best_tx_dft_beam_idx[val_idc]
rx_label_train,rx_label_val = best_rx_dft_beam_idx[train_idc],best_rx_dft_beam_idx[val_idc]

torch_h_train,torch_h_val = torch.from_numpy(h_train).to(device),torch.from_numpy(h_val).to(device)
torch_tx_label_train,torch_tx_label_val = torch.from_numpy(tx_label_train).to(device),torch.from_numpy(tx_label_val).to(device)
torch_rx_label_train,torch_rx_label_val = torch.from_numpy(rx_label_train).to(device),torch.from_numpy(rx_label_val).to(device)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_h_train,torch_tx_label_train,torch_rx_label_train)
val = torch.utils.data.TensorDataset(torch_h_val,torch_tx_label_val,torch_rx_label_val)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = len(val), shuffle = False)
    
num_antenna_Tx = np.prod(parameters['bs_antenna'][0]['shape'])
num_antenna_Rx = np.prod(parameters['ue_antenna']['shape'])

print('num probing beams = {}.'.format(n_probing_beam))

# NN implementing the joint Tx and Rx beamforming signal chain, including the probing beams and the beam synthesizer
autoencoder = Joint_BF_Classifier(num_antenna_Tx = num_antenna_Tx, num_antenna_Rx = num_antenna_Rx,
                            num_probing_beam_Tx = n_probing_beam, num_probing_beam_Rx = n_probing_beam, 
                            num_Tx_beams = int(16*16), num_Rx_beams = int(8*8),
                            noise_power=measurement_noise_power, norm_factor=norm_factor).to(device)

model_setup_params = ("{}_BS_{}_CB_"+
                   "UE_rot_{}_"+
                   "{}_{}_probe_"+
                   "{}_FB_{}_"+
                   "{}_"+
                   "{}_gamma_{}_"+
                   "train_noise_{}_dBm_"+
                   "meas_gain_{}_"+
                   "IA_thresh_{}_"+
                   "h_NMSE_{}").format(parameters['scenario'], parameters['active_BS'][0],
                                             args.random_UE_rotation,
                                             n_probing_beam, args.learned_probing,                                        
                                             args.feedback_mode, num_feedback,
                                             args.beam_synthesizer,
                                             'CE', None,
                                             noise_power_dBm,
                                             measurement_gain,
                                             args.IA_threshold,
                                             args.h_NMSE_dB)

model_savefname = model_savefname_prefix+model_setup_params+".pt"

autoencoder_opt = optim.Adam(autoencoder.parameters(),lr=0.001, betas=(0.9,0.999), amsgrad=True)

t_start = time.time()
train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = fit_CB(autoencoder, train_loader, val_loader, autoencoder_opt, nn.CrossEntropyLoss(), nepoch, model_savefname = train_hist_savefname_prefix+model_setup_params, h_val = h[val_idc], h_NMSE_dB = args.h_NMSE_dB,device=device)  
t_end = time.time()

print('time to train {} epochs with batch size {}: {:.5f} min'.format(nepoch,batch_size,(t_end-t_start)/60))
autoencoder = autoencoder.cpu()
torch.save(autoencoder.state_dict(),model_savefname)

loss_hist_savefname = train_hist_savefname_prefix+model_setup_params+'.npy'
loss_hist = {}
loss_hist['Tx loss train'] = train_loss_hist[0]
loss_hist['Rx loss train'] = train_loss_hist[1]
loss_hist['Tx loss val'] = val_loss_hist[0]
loss_hist['Rx loss val'] = val_loss_hist[1]
loss_hist['Tx acc train'] = train_acc_hist[0]
loss_hist['Rx acc train'] = train_acc_hist[1]
loss_hist['Tx acc val'] = val_acc_hist[0]
loss_hist['Rx acc val'] = val_acc_hist[1]

np.save(loss_hist_savefname,loss_hist,allow_pickle=True)

fig, axs = plt.subplots(2, 2, figsize=(10,8))
axs[0, 0].plot(loss_hist['Tx loss train'], label='training')
axs[0, 0].plot(loss_hist['Tx loss val'], label='validation')
axs[0, 0].set_title('Tx loss')
axs[0, 0].set_xlabel('epoch')
axs[0, 1].plot(loss_hist['Rx loss train'], label='training')
axs[0, 1].plot(loss_hist['Rx loss val'], label='validation')
axs[0, 1].set_title('Rx loss')
axs[0, 1].set_xlabel('epoch')
axs[1, 0].plot(loss_hist['Tx acc train'], label='training')
axs[1, 0].plot(loss_hist['Tx acc val'], label='validation')
axs[1, 0].set_title('Tx acc')
axs[1, 0].set_xlabel('epoch')
axs[1, 1].plot(loss_hist['Rx acc train'], label='training')
axs[1, 1].plot(loss_hist['Rx acc val'], label='validation')
axs[1, 1].set_title('Rx acc')
axs[1, 1].set_xlabel('epoch')
fig.tight_layout()
fig.savefig(train_hist_savefname_prefix+model_setup_params+'.pdf',dpi=300,format='pdf')