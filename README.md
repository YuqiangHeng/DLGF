# Grid-Free MIMO Beam Alignment through Site-Specific Deep Learning
### This repo contains the code to implement the site-specific deep learning (DL) pipeline for mmWave beam alignment proposed in the paper: 
> [**Y. Heng and J. G. Andrews, "Grid-Free MIMO Beam Alignment through Site-Specific Deep Learning," in IEEE Transactions on Wireless Communications, Jun. 2023, early access.**](https://ieeexplore.ieee.org/abstract/document/10151679) <br/>
#### Here's the [**Arxiv Version**](https://arxiv.org/abs/2102.08579). <br/>
### The conference version of this work won the **Best Paper Award** at the 2022 IEEE IEEE Global Communications Conference (GLOBECOM) in the SAC-MLC track:
> [**Y. Heng and J. G. Andrews, "Grid-less mmWave Beam Alignment through Deep Learning," in Proc. IEEE GLOBECOM, Dec. 2022.**](https://ieeexplore.ieee.org/document/10001720) <br/>

## Overview
### 1. Install the required python version and required packages (see requirements.txt)
### 2. Download the [**DeepMIMO dataset**](https://deepmimo.net/) and place the scenario files in the in your desired directory.
### 3. Train models: run the DLGF_train.py file to train grid-free (GF) models, and run the DLCB_train_grid.py file to train codebook-based (CB) models.
### 4. Compile and plot results: use the compile_results.ipynb notebook to compile and plot the results, use the visualize_leanred_beams.ipynb notebook to plot the learned probing/sensing beams as well as the predicted beams.

## Model Training
### Train GF or CB models: run the DLGF_train.py/DLCB_train.py file to train models. The following parameters can be adjusted in the argument:
- **scenario**: DeepMIMO Ray-tracing scenario: O1_28, I3_60, O1_28B, Boston_5G, etc.
- **activated_BS**: index of the activated BS in the DeepMIMO scenario 
- **num_probing_beam**: number of probing beams 
- **array_type**: BS and UE array type, ULA or UPA 
- **BW**: bandwidth in MHz 
- **noise_PSD_dB**: noise PSD in dBm/Hz 
- **measurement_gain**: spreading gain of the probing measurements 
- **h_NMSE_dB**: channel estimation error in dB, to generate noisy training data 
- **IA_threshold**: IA threshold in dB, UE need to achieve highe SNR than this threshold with one of the probing beams for initial access 
- **UE_rotation**: randomly rotate the UE antenna array 
- **use_specific_Tx_power**: use specific Tx power instead of the default value 
- **Tx_power_dBm**: Tx power in dBm 
- **num_feedback**: number of probing beam measuements to feed back to the BS, None means all the measurements are fed back
- **feedback_mode**: feedback mode for the probing measurements: diagonal, full or max. Consider measurements of all Tx/Rx beam pairs as a square matrix, feed back diagonal elements (each probing beam is matched to a sensing beam), the full matrix (measure all combinations), or the max element for each transmit beam (feedback the best sensing beam for each probing beam)
- **beam_synthesizer**: beam prediction function architecture, MLP or CNN
- **learned_probing**: TxRx: learn Tx and Rx probing beams jointly; Tx: learn Tx probing beams only; Rx: learn Rx probing beams only. If only the probing/sensing beams are learned, the sensing/probing beams on the other side are sub-sampled DFT beams.
- **gamma**: γ is the weight of the normalized BF loss, 1-gamma is the weight of the IA loss 
- **loss_fn**: loss function for training the GF models: BF_IA_loss, SPE_loss, BF_loss. BF_IA_loss is the proposed loss function in the paper that combines the normalized BF gain of the predicted beams and the gain of the probing beams. SPE_loss is based on the achievable rate with the predicted Tx and Rx beams. BF_loss is the unnormalized BF gain of the predicted beams.
- **nepoch**: number of epochs to train for 
- **batch_size**: batch size 
- **dataset_split_seed**: random seed for partitioning the dataset into training and testing 
#### For example, to train a GF model for the O1_28 scenario with random UE rotations, using 16 probing beams, noisy training data with NMSE -11 dB and optimized for the BF loss, run the following command:
```python
python DLGF_train.py --scenario O1_28 --num_probing_beam 16 --h_NMSE_dB -11 --UE_rotation --feedback_mode diagonal --beam_synthesizer MLP --learned_probing TxRx --loss_fn BF_loss --nepoch 2000 --batch_size 256
```
#### To train a CB model with the same setting, run the following command:
```python
python DLCB_train.py --scenario O1_28 --num_probing_beam 16 --h_NMSE_dB -11 --UE_rotation --feedback_mode diagonal --beam_synthesizer MLP --learned_probing TxRx --nepoch 2000 --batch_size 256
```
Note: The latest O1_28 dataset seems to be slight different from the version I downloaded in 2022. You may need to tune the Tx power and the spreding gain (--measurement_gain) accordingly to get a reasonable SNR range. The version of dataset that I used has been uploaded [**here**](https://www.dropbox.com/scl/fi/w8wvcpacl36pwca20gj6o/O1_28.zip?rlkey=wvld0pprm6rjk4ll8gk9d6y7o&dl=0). 
## Compiling and Plotting Results
### The notebook compile_results.ipynb contains the code to reproduce the plots and results in the paper. After training the models, simply set the scenario to one of the DeepMIMO scenarios and run the notebook. Some pre-trained models are included in the /Models folder.

### Enjoy the code and please cite our work if you find it useful for your research! 
### Feel free to contact me at yuqiang.heng@utexas.edu for any question or if you would like to discuss.




