import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

class BirdSongDataset(Dataset):
    def __init__(self, 
                    type, 
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        if type == "freefield1010":
            annotations_file="/speech/Databases/Birdsong/BirdSong/freefield1010_data.csv"
        elif type == "Warblr":
            annotations_file="/speech/Databases/Birdsong/BirdSong/Warblr_data.csv"
        elif type == "combined":
            annotations_file="/speech/Databases/Birdsong/BirdSong/Warblr_data.csv"
        else :
            raise NotImplementedError    
        self.uttr_labels= pd.read_csv(annotations_file)
        self.feat_root = "/speech/Databases/Birdsong/BirdSong/"
        self.transform = transform
        self.sample_rate = sample_rate
        self.no_of_classes=2

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        path,label = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.feat_root,path)
        uttr_melspec = np.load(uttr_path)
        return uttr_melspec, label

