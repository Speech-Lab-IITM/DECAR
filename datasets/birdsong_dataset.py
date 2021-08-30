import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os



class BirdSongDataset(Dataset):
    def __init__(self, annotations_file="./birdsong/6035814", 
                    uttr_dir = "./birdsong/wav/",
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        self.uttr_labels= pd.read_csv(annotations_file)
        self.uttr_dir = uttr_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.extract_mffc = torchaudio.transforms.MFCC(
                                            sample_rate=self.sample_rate,
                                            n_mfcc=30,
                                            log_mels=True) 

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        idx,label = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.uttr_dir, str(idx)+".wav")
        uttr_mfcc = self.read_mfcc(uttr_path)
        return uttr_mfcc, label

    def read_mfcc(self,filename):
        waveform, sample_rate = torchaudio.load(filename)   
        return torch.transpose(self.extract_mffc(waveform),-2,-1) # change shape -> C,T,nfeats
