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
    def __init__(self, type, use_l2 ,
                    transform=None,
                    target_transform=None,
                    sample_rate=16000):
        self.root_dir =DataUtils.root_dir["Libri100"] 
        if type == "train":
            annotations_file=os.path.join(self.root_dir,"train_data"+use_l2+".csv")
        elif type == "test":
            annotations_file=os.path.join(self.root_dir,"test_data"+use_l2+".csv")    
        else:
            raise NotImplementedError
        self.uttr_labels= pd.read_csv(annotations_file)
        self.transform = transform
        self.sample_rate = sample_rate
        self.train_labels = pd.read_csv(os.path.join(self.root_dir,"train_data"+use_l2+".csv"))
        self.labels_dict = DataUtils.map_labels(self.train_labels['Label'].to_numpy())
        self.no_of_classes= len(self.labels_dict)

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        idx,label = self.uttr_labels.iloc[idx,:]
        uttr_path = os.path.join(self.root_dir, str(idx)+".wav.npy")
        uttr_melspec = np.load(uttr_path)
        return uttr_melspec, self.labels_dict[label]

