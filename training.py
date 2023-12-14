import dac
import os
from dataset import data_loader_speech_train, data_loader_noise
import torchaudio.functional as F
import torch
import numpy as np
from audiotools import AudioSignal
from torchmetrics.audio import SignalNoiseRatio
import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="02456_deep_learning",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.00001,
    "architecture": "DAC",
    "dataset": "Audio-DTU",
    "epochs": 10,
    "loss": "MSE",
    }
)


# from torchmetrics.audio import SignalNoiseRatio
snr = SignalNoiseRatio().to('cuda')
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
epochs = 10

idy, noise_test = (next(enumerate(data_loader_noise)))
idy, speech_test = (next(enumerate(data_loader_speech_train)))
speech_test, sample_rate = speech_test
noise_test, _ = noise_test

signal_test = F.add_noise(speech_test, noise_test, snr=torch.tensor([[2.0]]))
signal_test = AudioSignal(signal_test, sample_rate=sample_rate)
signal_test.to('cuda')

#Use one noice for now
idy, noise = (next(enumerate(data_loader_noise))) #a random item gets removed when next is called .. i think    

tst_cnt = 0
#Epochs
for i in range(epochs):
    g_loss = 0
    g_metrics = 0

    for idx, speech in enumerate(data_loader_speech_train):
        # idy, noise = (next(enumerate(data_loader_noise))) #a random item gets removed when next is called .. i think        

        #Get sounds and sample_rates
        speech, sample_rate = speech
        noise, _ = noise
        # noise.to('cuda')
        
        # Combine noise, speech
        signal = F.add_noise(speech,noise, snr=torch.tensor([[2.0]]))
        signal = AudioSignal(signal, sample_rate=sample_rate)
        if tst_cnt == 0:
            signal.write(f'input_sounds/input_test{tst_cnt}.wav')                
            tst_cnt += 1
        signal.to('cuda')
        speech = speech.to('cuda')
        
        #Use Decript Model To encode
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = model.encode(x) 
        y = model.decode(z)        
        
        #Loss SNR here
        y = y[:, :, :speech.size(2)]
        
        #MSE LOSS, why does the mse_loss have a backwards function, are we using 2 optimizers right now?
        loss = torch.nn.functional.mse_loss(y, speech)
        # loss = 1/snr(y, speech)


        #Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics = snr(y, speech)



        #Loss output
        '''y_audio_signal = AudioSignal(y.cpu().detach().numpy(), sample_rate=sample_rate)
        y_audio_signal.write(f"output{i}.wav")'''
        # print(f'loss: {loss}')
        # print(f'snr: {metrics}')
        g_loss += loss
        g_metrics += metrics
    
    
    print(f'epoch: {i}\t loss: {g_loss} \t metric: {g_metrics}')
    wandb.log({"snr": g_metrics, "loss": g_loss})
    
    #validation set here


    #save model
    if i == (epochs-1):
        #Use model for audio file
        x = model.preprocess(signal_test.audio_data, signal_test.sample_rate)
        z, codes, latents, _, _ = model.encode(x) 
        
        y = model.decode(z)

        #Be able to hear file
        y = y[:, :, :speech.size(2)]
        y_numpy = y.cpu().detach().numpy()
        y_normalized = np.clip(y_numpy, -1.0, 1.0)  
        y_audio_signal = AudioSignal(y_normalized, sample_rate=sample_rate)

        #to Cpu
        
        #Don't know why input_sounds isn't writing to file
        print("outputting to file")
        y_audio_signal.write(f"output_sounds/output.wav") 

        signal_test.to('cpu')

        signal_test.write(f'input_sounds/input.wav')
        wandb.finish()

        
