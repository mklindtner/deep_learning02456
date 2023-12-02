import dac
import os
from dataset import data_loader_speech_train, data_loader_noise
import torchaudio.functional as F
import torch
import numpy as np
from audiotools import AudioSignal
from torchmetrics.audio import SignalNoiseRatio
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
#Epochs
for i in range(epochs):
    for idx, speech in enumerate(data_loader_speech_train):
        idy, noise = (next(enumerate(data_loader_noise))) #a random item gets removed when next is called .. i think        

        #Get sounds and sample_rates
        speech, sample_rate = speech
        noise, _ = noise
        # speech_element = speech[torch.arange(speech.size(0)), 0]
        # noise_element = noise[torch.arange(speech.size(0)), 0]
        
        signal = F.add_noise(speech,noise, snr=torch.tensor([[2.0]]))

        # Combine noise and speech for the training set element
        # signal = F.add_noise(speech_element, noise_element, snr=torch.tensor([[1.5]]))     

        signal = AudioSignal(signal, sample_rate=sample_rate)
        #test signal.sample_rate vs sample_rate

        signal.to('cuda')
        speech = speech.to('cuda')
        #Use Decript Model To encode
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        #Generate a batch size of (1,y,z)
        # x = x[None, :, :]
        # x[:,0,0] = 1
        
        z, codes, latents, _, _ = model.encode(x) 
        # print(z.shape)       
        
        #label - Doesn't work right now
        y = model.decode(z)        
        #Loss SNR heremsd
        y = y[:, :, :speech.size(2)]
        loss = torch.nn.functional.mse_loss(y, speech)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics = snr(y, speech)
        '''y_audio_signal = AudioSignal(y.cpu().detach().numpy(), sample_rate=sample_rate)
        y_audio_signal.write(f"output{i}.wav")'''
        print(f'loss: {loss}')
        print(f'snr: {metrics}')
    print("Epoch: ", i)
    x = model.preprocess(signal_test.audio_data, signal_test.sample_rate)
    z, codes, latents, _, _ = model.encode(x) 
    y = model.decode(z)
    y = y[:, :, :speech.size(2)]
    y_numpy = y.cpu().detach().numpy()
    y_normalized = np.clip(y_numpy, -1.0, 1.0)  
    y_audio_signal = AudioSignal(y_normalized, sample_rate=sample_rate)
    print("OUTPUT")
    y_audio_signal.write(f"outpuut{i}.wav")
    speech_audio_signal = AudioSignal(speech_test, sample_rate=sample_rate)
    speech_audio_signal.write(f"inpuut{i}.wav")
