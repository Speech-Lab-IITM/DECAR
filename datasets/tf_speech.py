import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils


class TfSpeech(Dataset):
    def __init__(self, annotations_file="/speech/sandesh/icassp/tf_speech/train/labels.csv",
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        self.uttr_df= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.uttr_df)

    def __getitem__(self, idx):
        audio_path,label = self.uttr_df.iloc[idx,:]
        uttr_mfcc = DataUtils.read_mfcc(audio_path)
        return uttr_mfcc, label