from datasets.voxceleb import Voxceleb1Dataset
from datasets.birdsong_dataset import BirdSongDataset ,BirdSongDatasetL2
from datasets.libri100 import Libri100 ,Libri100L2
from datasets.tut_urban_sounds import TutUrbanSounds ,TutUrbanSoundsL2
from datasets.musical_instruments import MusicalInstrumentsDataset
from datasets.iemocap import IEMOCAPDataset , IEMOCAPDatasetL2
from datasets.speech_commands_v1 import SpeechCommandsV1 
from datasets.speech_commands_v2 import SpeechCommandsV2 , SpeechCommandsV2L2
import torch

def get_dataset(downstream_task_name):
    if downstream_task_name == "birdsong_freefield1010":
        return split_dataset(BirdSongDataset(type="freefield1010"))
    elif downstream_task_name == "birdsong_warblr":
        return split_dataset(BirdSongDataset(type="Warblr"))
    elif downstream_task_name == "birdsong_combined":
        return split_dataset(BirdSongDataset(type="combined")) 
    elif downstream_task_name == "birdsong_combined_l2":
        return split_dataset(BirdSongDatasetL2(type="combined"))        
    elif downstream_task_name == "speech_commands_v1":
        return SpeechCommandsV1() , SpeechCommandsV1()
    elif downstream_task_name == "speech_commands_v2":
        return SpeechCommandsV2(type="train") , SpeechCommandsV2(type="test") 
    elif downstream_task_name == "speech_commands_v2_l2":
        return SpeechCommandsV2L2(type="train") , SpeechCommandsV2L2(type="test")     
    elif downstream_task_name == "libri_100":
        return Libri100(type="train") , Libri100(type="test") 
    elif downstream_task_name == "libri_100_l2":
        return Libri100L2(type="train") , Libri100L2(type="test")     
    elif downstream_task_name == "musical_instruments":
        return MusicalInstrumentsDataset(type="train") , MusicalInstrumentsDataset(type="test")
    elif downstream_task_name == "iemocap":
        return IEMOCAPDataset(type='train'),IEMOCAPDataset(type='test')    
    elif downstream_task_name == "iemocap_l2":
        return IEMOCAPDatasetL2(type='train'),IEMOCAPDatasetL2(type='test')        
    elif downstream_task_name == "tut_urban": 
        return TutUrbanSounds(type="train"),TutUrbanSounds(type="test")
    elif downstream_task_name == "tut_urban_l2":
        return TutUrbanSoundsL2(type="train"),TutUrbanSoundsL2(type="test")    
    elif downstream_task_name == "voxceleb_v1":
        return Voxceleb1Dataset(type="train") , Voxceleb1Dataset(type="test")   
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