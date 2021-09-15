import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

class TutUrbanSounds(Dataset):
    '''
    audio_root: /speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/audio
    '''
    def __init__(self, type="train" , 
                    audio_root = "/speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/audio",
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        if(type == "train"):
            annotations_file = "/speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/evaluation_setup/train_data.csv"
        elif(type=="valid"):
            annotations_file = "/speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/evaluation_setup/valid_data.csv"
        else:
            raise NotImplementedError    
        self.uttr_labels= pd.read_csv(annotations_file)
        self.audio_root = audio_root
        self.transform = transform
        self.sample_rate = sample_rate
        self.no_of_classes= 585#
        self.labels_dict = DataUtils.map_labels(self.uttr_labels['Label'].to_numpy())

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        idx,label = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.audio_root, str(idx)+".wav.npy")
        # uttr_mfcc = DataUtils.read_mfcc(uttr_path)
        # uttr_melspec = DataUtils.extract_log_mel_spectrogram(uttr_path)
        uttr_melspec = np.load(uttr_path)
        return uttr_melspec, self.labels_dict[label]

