import dac
import os
from dataset import data_loader_speech_train, dataset_noise, data_loader_speech_test,data_loader_speech_val
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

snr = SignalNoiseRatio().to('cuda')
si_snr = ScaleInvariantSignalNoiseRatio().to('cuda')

model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.load_state_dict(torch.load("model_weights_mae.pth"))
model.to('cuda')

# model_path = dac.utils.download(model_type="44khz")
# model = dac.DAC.load(model_path)
# model.load_state_dict(torch.load("model_weights_mse.pth"))
# model.to('cuda')

# model_path = dac.utils.download(model_type="44khz")
# model = dac.DAC.load(model_path)
# model.load_state_dict(torch.load("model_weights_si_snr.pth"))
# model.to('cuda')

# model_path = dac.utils.download(model_type="44khz")
# model = dac.DAC.load(model_path)
# model.load_state_dict(torch.load("model_weights_snr.pth"))
# model.to('cuda')


epochs = 10


squim_objective_model = SQUIM_OBJECTIVE.get_model().to('cuda')
squim_subjective_model = SQUIM_SUBJECTIVE.get_model().to('cpu')

g_loss_mae = 0
g_loss = 0
g_metrics = 0
g_stoi = 0
g_pesq = 0
g_si_sdr = 0
g_mos = 0
g_si_snr = 0
g_snr = 0
leng = len(data_loader_speech_val)
print(f"leng: {leng}")


with torch.no_grad():
    model.eval()
    # test_accuracies = []
    for idx, speech in enumerate(data_loader_speech_val):
            random = np.random.randint(0, 100)
            noise, _= dataset_noise[random]

            #Get sounds and sample_rates
            speech, sample_rate = speech
            noise = noise.unsqueeze(0)
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

            #Change y_orig to be the same size as speech
            y = y_orig[:, :, :speech.size(2)]

            y = signal.audio_data
            
            #LOSS
            loss = mse_loss(y, speech)
            loss2 = mae_loss(y, speech)

            #Metrics           
            g_loss += loss
            g_loss_mae += loss2
            g_si_snr += si_snr(y, speech)
            g_snr += snr(y, speech)
            # print(f"g_snr: {g_snr}\t g_loss_mse: {g_loss}")

            # Match y sample rate to squim model sample rate
            if sample_rate != 16000:
                y_resampled = F.resample(y_orig, orig_freq=sample_rate, new_freq=16000)
            else:
                y_resampled = y_orig

            #paper: https://arxiv.org/pdf/2304.01448.pdf
            # Calculate squim metrics

            #Make the y_resampled into an appropriate tensor for squim
            y_resampled.to('cuda')

            #objective model metrics squim
            stoi, pesq, si_sdr = squim_objective_model(y_resampled[0])
            g_stoi += stoi[0]
            g_pesq += pesq[0]
            g_si_sdr += si_sdr[0]          

            #subjective model metrics squim
            #Use CPU because we run out of memory on the GPU
            speech_cpu = speech[0].to('cpu')
            y_resampled_cpu = y_resampled[0].to('cpu')
            mos = squim_subjective_model(y_resampled_cpu, speech_cpu)  
            g_mos += mos[0]
            if idx % 19 == 0:
                print(f"file_number: {idx+1}")

    print(f"<<<Average Metrics>>>")
    print(f"objective model metrics\n stoi_avg: {g_stoi/len(data_loader_speech_test):.6f}\t pesq_avg: {g_pesq/len(data_loader_speech_test):.6f}\t si_sdr_avg: {g_si_sdr/len(data_loader_speech_test):.6f}")
    print(f"subjective model metrics\n mos: {g_mos/len(data_loader_speech_test):.6f}")
    print(f"mse loss average: {g_loss/len(data_loader_speech_test):.6f}\t loss_mae average: {g_loss_mae/len(data_loader_speech_test):.6f}\t")
    print(f"snr: {g_snr/len(data_loader_speech_test):.6f}\t si_snr:{g_si_snr/len(data_loader_speech_test):.4f}\t sdr: {g_si_sdr/len(data_loader_speech_test):.6f}")

