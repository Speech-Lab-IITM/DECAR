from datasets.birdsong_dataset import BirdSongDataset
from datasets.libri100 import Libri100
import torch

def get_dataset(downstream_task_name):
    if downstream_task_name == "birdsong_freefield1010":
        return split_dataset(BirdSongDataset(type="freefield1010"))
    elif downstream_task_name == "birdsong_warblr":
        return split_dataset(BirdSongDataset(type="Warblr"))
    elif downstream_task_name == "birdsong_combined":
        return split_dataset(BirdSongDataset(type="combined"))    
    elif downstream_task_name == "speech_commands_v1":
        raise NotImplementedError
    elif downstream_task_name == "speech_commands_v2":
        raise NotImplementedError 
    elif downstream_task_name == "libri_100":
        return Libri100(type="train") , Libri100(type="valid") 
    elif downstream_task_name == "musical_instruments":
        raise NotImplementedError
    elif downstream_task_name == "iemocap":
        raise NotImplementedError    
    elif downstream_task_name == "tut_urban":
        raise NotImplementedError 
    elif downstream_task_name == "voxceleb1":
        raise NotImplementedError    
    elif downstream_task_name == "musan":
        raise NotImplementedError                   
    else:
        raise NotImplementedError


def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset,valid_dataset