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
import numpy as np

class NoiseSpeechDataset(Dataset):
    def __init__(self, directory, target_sample_rate, len_speech, is_speech, dataset_type):
        self.target_sample_rate = target_sample_rate
        self.directory = directory
        self.len_speech = len_speech
        self.files_ns = os.listdir(directory)
        self.is_speech = is_speech
        self.type = dataset_type

        #split the dataset into train, val and test
        self.initialize_dataset_state() 

    def initialize_dataset_state(self):
        if self.type == "noise":
            return

        #By using a seed we guarantee the shuffle is always the same for training, validation and test sets
        np.random.seed(6977) 
        np.random.shuffle(self.files_ns)
        
        # random.seed(6977)

        # Sizes for each part of the array
        size_train_data = int(len(self.files_ns) * 0.70)  # 70% of the total size
        size_val_data = int(len(self.files_ns) * 0.15)  # 15% of the total size
        size_test_data = int(len(self.files_ns) * 0.15)  # 15% of the total size
        

        #use a seed to shuffle - making each split identitical each time the dataset is initialized
        split_train = int(size_train_data)
        split_val = int(size_val_data) + split_train
        split_test = int(size_test_data) + split_val + split_train

        #Splits up the entire clean sound directory into three parts
        self.indicies_train = self.files_ns[0:split_train]
        self.indicies_val = self.files_ns[split_train:split_val]
        self.indicies_test = self.files_ns[split_val:split_test]

    def __len__(self):
        return len(self.files_ns)

    def __getitem__(self, idx):         
        file_path = None
        
        if self.type == "train":            
            item = random.choice(self.indicies_train)
            file_path = os.path.join(self.directory, item)            
        elif self.type == "val":
            item = random.choice(self.indicies_val)            
            file_path = os.path.join(self.directory, item)            
        elif self.type == "test":
            item = random.choice(self.indicies_test)
            file_path = os.path.join(self.directory, item)            
        else: #our local noise directory
            file_path = os.path.join(self.directory, self.files_ns[idx]) 
        

        waveform, sample_rate = torchaudio.load(file_path,backend="soundfile")
        # print("filepath:", file_path)
        # print(waveform.shape)
        seconds = waveform.shape[1]/sample_rate
        # f'seconds:{seconds}'
        # print(seconds)
        # print("-----")
        if self.is_speech:
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

#Paths for HPC-full-dataset
dataset_clean_speech_full_path = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/read_speech"

# Create the dataset and data loader
target_sample_rate = 44100
len_speech = 5

#Dataset
dataset_speech_train = NoiseSpeechDataset(dataset_clean_speech_full_path, target_sample_rate, len_speech, is_speech=True, dataset_type="train")
dataset_speech_test = NoiseSpeechDataset(dataset_clean_speech_full_path, target_sample_rate, len_speech, is_speech=True, dataset_type="validation")
dataset_noise =  NoiseSpeechDataset(dataset_noise_path, target_sample_rate, len_speech, False, dataset_type="noise")

#Dataset / Database - old dataset
# dataset_speech_train = NoiseSpeechDataset(dataset_speech_path, target_sample_rate, len_speech, is_speech=True, dataset_type="train")
# dataset_speech_test = NoiseSpeechDataset(dataset_speech_test_path, target_sample_rate, len_speech, is_speech=True, dataset_type="test")
# dataset_noise =  NoiseSpeechDataset(dataset_noise_path, target_sample_rate, len_speech, False, dataset_type="noise")

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