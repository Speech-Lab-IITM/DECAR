import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os

from datasets.data_utils import DataUtils

class IEMOCAPDataset(Dataset):
    def __init__(self,  
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):        
        self.feat_root =  DataUtils.root_dir["IEMOCAP"]
        annotations_file = os.path.join(self.feat_root,'data.csv')
        self.uttr_labels= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
        self.labels_dict = DataUtils.map_labels(self.uttr_labels['Label'].to_numpy())
        self.no_of_classes= len(self.labels_dict)

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        path,label = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.feat_root,path)
        uttr_melspec = np.load(uttr_path)
        return uttr_melspec, self.labels_dict[label]

