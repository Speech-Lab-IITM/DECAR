import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

# stats
# train total : 064722
# test  total : 158539

class SpeechCommandsV1(Dataset):
    def __init__(self, annotations_file="/speech/sandesh/icassp/tf_speech/train/labels.csv",
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        self.uttr_df= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
        
        test_labels = ["yes", "no", "up", "down","left", "right", "on", "off", "stop", "go" ] #10
        core_labels = test_labels + ["zero","one","two","three","four","five","six","seven","eight","nine"] # 20
        auxliary_labels = ["bird","dog","happy", "wow","bed","cat","house","marvin","sheila","tree"] #30
        self.labels = core_labels + auxliary_labels
        self.no_of_classes=len(self.labels) # 30

    def __len__(self):
        return len(self.uttr_df)

    def get_label_id(self,label):
        try:
            label_id = self.labels.index(label)
        except ValueError as e:
            label_id = len(self.labels)  
        return label_id 

    def __getitem__(self, idx):
        audio_path,label = self.uttr_df.iloc[idx,:]
        #uttr_mfcc = DataUtils.read_mfcc(audio_path)
        uttr_melspec = DataUtils.extract_log_mel_spectrogram(audio_path)
        return uttr_melspec, self.get_label_id(label)


class SpeechCommandsV1Test(Dataset):
    def __init__(self,annotations_file="/speech/sandesh/icassp/tf_speech/test/labels.csv",
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        self.uttr_df= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
        self.labels=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go' ]
        self.no_of_classes=len(self.labels)

    def __len__(self):
        return len(self.uttr_df)   

    def __getitem__(self, idx):
        audio_path,file_name = self.uttr_df.iloc[idx,:]
        #uttr_mfcc = DataUtils.read_mfcc(audio_path)
        uttr_melspec = DataUtils.extract_log_mel_spectrogram(audio_path)
        return uttr_melspec, file_name 

    def get_label(self,idx):
        try:
            return self.labels[idx]
        except Exception as e:
            return "unknown"
        