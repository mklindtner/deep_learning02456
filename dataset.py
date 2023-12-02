import torch
import os
import torchaudio
from pathlib import Path
import torchaudio.functional as F
import torchaudio.transforms as T
from audiotools import AudioSignal
# from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import os
import random
import soundfile
# p = Path('.')
# ns_Data = list(p.glob('noisespeech_pairs/*'))
# s_Data = list(p.glob('Speech/*'))

class NoiseSpeechDataset(Dataset):
    def __init__(self, directory, target_sample_rate, len_speech, is_speech):
        self.target_sample_rate = target_sample_rate
        self.directory = directory
        self.len_speech = len_speech
        self.files_ns = os.listdir(directory)
        self.is_speech = is_speech

    def __len__(self):
        return len(self.files_ns) #assumes noisespeech_pairs and speech have same length

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files_ns[idx])
        waveform, sample_rate = torchaudio.load(file_path,backend="soundfile")
        # print("filepath:", file_path)
        # print(waveform.shape)
        seconds = waveform.shape[1]/sample_rate
        # f'seconds:{seconds}'
        # print(seconds)
        # print("-----")
        if self.is_speech: #speech directory
            if seconds > 5: 
                random_start = random.randrange(0, waveform.shape[1]-sample_rate*self.len_speech)
                #print("Start:", random_start, " End: ",random_start+sample_rate*self.len_speech)
                waveform = waveform[:,random_start:random_start+sample_rate*self.len_speech]
            else: 
                # print("inside speech else")
                #padding here
                pad_len = len_speech*sample_rate - waveform.shape[1]
                # waveform = F.pad_waveform(waveform, 0, pad_len, "constant")
                num_samples_to_pad = max(0, pad_len)           
                z = torch.zeros(1,num_samples_to_pad) #waveform.shape[0]   
                waveform = torch.cat((waveform,z),1)                
        else: #noise directory
            if seconds > 5:
                random_start = random.randrange(0, waveform.shape[1]-sample_rate*len_speech)
                waveform = waveform[:,random_start:random_start+sample_rate*self.len_speech]            
            else:
                while(waveform.shape[1]<sample_rate*len_speech):
                    pad_len = len_speech*sample_rate - waveform.shape[1]
                    num_samples_to_pad = min(waveform.shape[1], pad_len)
                    # print("noise")
                    waveform = torch.cat((waveform,waveform[0:num_samples_to_pad]))
        
        # print("<---->")
        #print(waveform.shape[1]/sample_rate)

        #Resample here
        transform = T.Resample(sample_rate, self.target_sample_rate)    
        waveform = transform(waveform)
                
        return waveform, self.target_sample_rate

# Specify the directory containing your data
# dataset_directory = 'noisespeech_pairs'
dataset_noise_path = "Audio/Noise"
dataset_speech_path = "Audio/Speech"
dataset_speech_test_path = "Audio/Speech_test"


# Create the dataset and data loader
target_sample_rate = 44100
len_speech = 5

#Dataset / Database
dataset_speech_train = NoiseSpeechDataset(dataset_speech_path, target_sample_rate, len_speech, is_speech=True)
dataset_speech_test = NoiseSpeechDataset(dataset_speech_test_path, target_sample_rate, len_speech, is_speech=True)
dataset_noise =  NoiseSpeechDataset(dataset_noise_path, target_sample_rate, len_speech, False)

#Loader
g_bs = 1
data_loader_speech_train = DataLoader(dataset_speech_train, batch_size=g_bs, shuffle=True)
data_loader_speech_test = DataLoader(dataset_speech_test, batch_size=g_bs, shuffle=True)
data_loader_noise = DataLoader(dataset_noise, batch_size=g_bs, shuffle=True)

# print(f'{next(iter(data_loader_noise))}')
# foo = next(iter(data_loader_speech))
# bar = next(iter(data_loader_noise))
# print(foo)
# print(bar)
# print("---")
# print(bar)