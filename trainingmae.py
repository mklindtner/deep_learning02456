import dac
import os
from dataset import data_loader_speech_train, dataset_noise
from squim import si_snr
import torchaudio.functional as F
import torch
import numpy as np
from audiotools import AudioSignal
from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio
import wandb
import random
import itertools
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

print("starting training")
mae_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Real Model training ",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.00001,
    "architecture": "DAC",
    "dataset": "Audio-DTU",
    "epochs": 10,
    "loss": "MAE",
    }
)


# from torchmetrics.audio import SignalNoiseRatio
snr = SignalNoiseRatio().to('cuda')
si_snr = ScaleInvariantSignalNoiseRatio().to('cuda')
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
epochs = 10
#cycling_data_loader_noise = itertools.cycle(data_loader_noise)
#unsure
#idy, noise_test = (next(enumerate(data_loader_noise)))
idy, speech_test = (next(enumerate(data_loader_speech_train)))
speeech, sample_rate = speech_test
#noise, _ = noise_test
#print(next(enumerate(data_loader_noise)))

squim_objective_model = SQUIM_OBJECTIVE.get_model().to('cuda')
squim_subjective_model = SQUIM_SUBJECTIVE.get_model().to('cpu')
#Epochs
for i in range(epochs):
    g_loss = 0
    g_metrics = 0
    g_stoi = 0
    g_pesq = 0
    g_si_sdr = 0
    g_mos = 0
    print(len(data_loader_speech_train)) #7840
    leng = len(data_loader_speech_train)
    for idx, speech in enumerate(data_loader_speech_train):
        # print("started data_loader_noise")
        random = np.random.randint(0, len(dataset_noise))
        noise, _= dataset_noise[random]
        #print(noise)
        # print("passed data_loader_noise")
        #Get sounds and sample_rates
        speech, sample_rate = speech
        noise = noise.unsqueeze(0)
        #speech = speeech
        noise.to('cuda')
        # Combine noise, speech
        snr_dbs = torch.tensor([[2.0]])
        signal = F.add_noise(speech,noise, snr=snr_dbs)
        if torch.isnan(signal).any():
            print("NaN values detected in signal.audio_data, SKIPPING")
            continue
        signal = AudioSignal(signal, sample_rate=sample_rate)

        signal.to('cuda')
        speech = speech.to('cuda')
        
        #Use Decript Model To encode
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = model.encode(x)
        y_orig = model.decode(z)     
        # print(f"y_orig: {y_orig.shape}\t y_orig2: {y_orig[0].shape}")

        #Loss SNR here
        y = y_orig[:, :, :speech.size(2)]
        # print(f"y_orig: {y_orig.shape}\t y: {y.shape}")
        
        #LOSS
        #loss = mse_loss(y, speech)
        loss = mae_loss(y, speech)
        #loss = 1/snr(y, speech)
        #loss = 1/si_snr(y, speech)
        #Metrics from squim
        #print("SUBJECTIVE MODEL")
        # Match y sample rate to squim model sample rate
        #if sample_rate != 16000:
        #    y_resampled = F.resample(y_orig, orig_freq=sample_rate, new_freq=16000)
        #else:
        #    y_resampled = y_orig

        #paper: https://arxiv.org/pdf/2304.01448.pdf
        # Calculate squim metrics
        
        #Make the y_resampled into an appropriate tensor for squim
        #y_resampled.to('cuda')
        #print(f"speech: {speech[0].shape}\t y_resampled: {y_resampled[0].shape}")

        #objective model metrics
        #stoi_hyp, pesq_hyp, si_sdr_hyp = squim_objective_model(y_resampled[0])
        
        #print(f"signal-to-noise: {snr_dbs[0]}")
        #print(f"objective model metrics: stoi_hyp: {stoi_hyp[0]}\t pesq_hyp: {pesq_hyp[0]}\t si_sdr_hyp: {si_sdr_hyp[0]}")
        

        #subjective model metrics

        #Use CPU because we run out of memory on the GPU
        #speech_cpu = speech[0].to('cpu')
        #y_resampled_cpu = y_resampled[0].to('cpu')
        #loss = 1/squim_subjective_model(y_resampled_cpu, speech_cpu)        
        
        #print(f"subjective model metrics: mos: {mos[0]}")

        # see https://pytorch.org/audio/main/tutorials/squim_tutorial.html for usage of this last metric
        # print(f"si_snr: {squim_snr}")
        
        #Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     
        # if tst_cnt == 0:
        #     print(f"<breaking>")        
        #     break
        #     # signal.write(f'input_sounds/input_test2{tst_cnt}.wav')                
        #     # tst_cnt += 1

        #Loss output
        '''y_audio_signal = AudioSignal(y.cpu().detach().numpy(), sample_rate=sample_rate)
        y_audio_signal.write(f"output{i}.wav")'''
        print(loss)

        # print(f'loss: {loss}')
        # print(f'snr: {metrics}')
        g_loss += loss
        #g_metrics += metrics
        
        #Change these metrics as appropriate
        #g_stoi += stoi_hyp[0]
        #g_pesq += pesq_hyp[0]
        #g_si_sdr += si_sdr_hyp[0]
        #g_mos += mos[0]

        #Store squim_metrics heres
        #torch.cuda.empty_cache()
        # print("<finished a batch")
    print("<finished Epoch>")
    if i == (epochs-1):
        torch.save(model.state_dict(), 'model_weights.pth')
        wandb.save('model_weights.pth')
    #print(f'epoch: {i}\t avg loss: {g_loss/id} \t avg metric: {g_metrics/id}')
    #wandb.log({"snr": g_metrics/id, "loss": g_loss/id, "stoi": g_stoi/id, "pesq": g_pesq/id, "si_sdr": g_si_sdr/id, "mos": g_mos/id})
    print("loss:", g_loss)
    wandb.log({"avg loss": g_loss/leng})
    #validation set here

    
    #save model weights
    # if i == (epochs-1):
    #     #Use model for audio file
    #     x = model.preprocess(signal_test.audio_data, signal_test.sample_rate)
    #     z, codes, latents, _, _ = model.encode(x) 
        
    #     y = model.decode(z)

    #     #Be able to hear file
    #     y = y[:, :, :speech.size(2)]
    #     y_numpy = y.cpu().detach().numpy()
    #     y_normalized = np.clip(y_numpy, -1.0, 1.0)  
    #     y_audio_signal = AudioSignal(y_normalized, sample_rate=sample_rate)

    #     #to Cpu
    #     torch.save(model.state_dict(), 'model_weights.pth')
    #     wandb.save('model_weights.pth')
    #     print("outputting to file")
    #     y_audio_signal.write(f"output_sounds/output.wav") 

    #     signal_test.to('cpu')

    #     signal_test.write(f'input_sounds/input.wav')
    #     wandb.finish()

        
