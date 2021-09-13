import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

# ----Stats----
# speakers 585   
# test samples  :: 05708 
# train samples :: 22830 
# ---------------

class Libri100(Dataset):
    def __init__(self, type="train", 
                    audio_root = "/speech/Databases/Birdsong/Libri100SpkId/spec",
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        if type == "train":
            annotations_file="/speech/Databases/Birdsong/Libri100SpkId/train_data.csv"
        elif type == "valid":
            annotations_file="/speech/Databases/Birdsong/Libri100SpkId/test_data.csv"    
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

