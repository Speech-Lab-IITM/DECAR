from datasets.birdsong_dataset import BirdSongDataset
from datasets.libri100 import Libri100
from datasets.tut_urban_sounds import TutUrbanSounds
from datasets.musical_instruments import MusicalInstrumentsDataset
from datasets.iemocap import IEMOCAPDataset 
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
        return MusicalInstrumentsDataset(type="train") , MusicalInstrumentsDataset(type="valid")
    elif downstream_task_name == "iemocap":
        return split_dataset(IEMOCAPDataset())    
    elif downstream_task_name == "tut_urban":
        return TutUrbanSounds(type="train"),TutUrbanSounds(type="valid")
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
    train_dataset.no_of_classes = dataset.no_of_classes
    return train_dataset,valid_dataset