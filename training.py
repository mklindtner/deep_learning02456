import dac
import os
from dataset import data_loader_speech_train, data_loader_noise
import torchaudio.functional as F
import torch
from audiotools import AudioSignal
from torchmetrics.audio import SignalNoiseRatio
# from torchmetrics.audio import SignalNoiseRatio
snr = SignalNoiseRatio()
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
# model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
epochs = 1

#Epochs
for i in range(epochs):
    for idx, speech in enumerate(data_loader_speech_train):
        idy, noise = (next(enumerate(data_loader_noise))) #a random item gets removed when next is called .. i think        

        #Get sounds and sample_rates
        speech, sample_rate = speech
        noise, _ = noise
        # speech_element = speech[torch.arange(speech.size(0)), 0]
        # noise_element = noise[torch.arange(speech.size(0)), 0]
        
        signal = F.add_noise(speech,noise, snr=torch.tensor([[1.1]]))

        # Combine noise and speech for the training set element
        # signal = F.add_noise(speech_element, noise_element, snr=torch.tensor([[1.5]]))     

        signal = AudioSignal(signal, sample_rate=sample_rate)
        #test signal.sample_rate vs sample_rate

        # signal.to('cuda')
        #Use Decript Model To encode
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        #Generate a batch size of (1,y,z)
        # x = x[None, :, :]
        # x[:,0,0] = 1
        
        z, codes, latents, _, _ = model.encode(x) 
        # print(z.shape)       
        
        #label - Doesn't work right now
        y = model.decode(z)        
        print(y)
        #Loss SNR heremsd
        y = y[:, :, :speech.size(2)]
        loss = 1/snr(y,speech)
        print("loss:", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'loss: {loss}')
        if loss < 0.3:
            y_audio_signal = AudioSignal(y.cpu().detach().numpy(), sample_rate=sample_rate)
            y_audio_signal.write("output.wav")
            speech_audio_signal = AudioSignal(signal.cpu().detach().numpy(), sample_rate=sample_rate)
            speech_audio_signal.write("input.wav")

        if idx % 50 == 0:
            # print("test")
            # print(signal.data)
            # print(y)
            # print(f'x:{x} \n {x.shape} \n y:{y} \n {y.shape} \n --- \n ')
            
            # print(f'{signal.shape}')
            # print(noise_item)
            # print(f'noise_batch_len: {len(noise_item)} \t speech_batch_len: {len(item)})')            
            # print(f'item: {speech[torch.arange(speech.size(0)), 0].shape}')
            # print(noise.shape)
            # print(speech.shape)            
            # print(speech)
            print("----")
            # print(noise)
            # print(f'item {signal} \n')
        #Combine speech and noise