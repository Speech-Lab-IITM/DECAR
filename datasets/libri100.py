import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils


# 585 speakers
# 5708 test samples
# 22830 train samples


class Libri100(Dataset):
    '''
    /speech/ashish/data_wav/data_16000
    /speech/srayan/icassp/downstream/data/libri100_spkid
    train_split.txt
    '''
    def __init__(self, annotations_file="./birdsong/6035814", 
                    audio_root = "/speech/ashish/data_wav/data_16000",
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        self.uttr_labels= pd.read_csv(annotations_file)
        self.audio_root = audio_root
        self.transform = transform
        self.sample_rate = sample_rate
        self.no_of_classes= 585#

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        idx,label = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.audio_root, str(idx)+".wav")
        #uttr_mfcc = DataUtils.read_mfcc(uttr_path)
        uttr_melspec = DataUtils.extract_log_mel_spectrogram(uttr_path)
        return uttr_melspec, label

