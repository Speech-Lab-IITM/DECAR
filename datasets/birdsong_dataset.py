import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

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

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        idx,label = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.uttr_dir, str(idx)+".wav")
        uttr_mfcc = DataUtils.read_mfcc(uttr_path)
        return uttr_mfcc, label

