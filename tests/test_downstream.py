import torch
import pytest
import numpy as np

from efficientnet.model import DeepCluster_ICASSP
from datasets.birdsong_dataset import BirdSongDataset
from datasets.libri100 import Libri100
from datasets.iemocap import IEMOCAPDataset
from datasets.musical_instruments import MusicalInstrumentsDataset
from datasets.tut_urban_sounds import TutUrbanSounds
from datasets.data_utils import DataUtils

# test using :  python -m pytest tests/

def test_map_labels():
    map_dict = DataUtils.map_labels(np.array(['1','2','1']))
    assert len(map_dict.keys()) ==2


def test_model_ouputs():
    deep_model = DeepCluster_ICASSP(2)
    forward = deep_model(torch.randn(10,1,200,40))
    assert forward.shape == (10,2)

#-----------------------------BirdSong------------------#
def test_birdsong_dataset_single():
    dataset = BirdSongDataset(type='combined')
    mfcc = dataset.__getitem__(1784)
    assert mfcc[0].shape ==(998,64)
    


def test_birdsong_dataset_batch():
    dataset = BirdSongDataset(type='combined')
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=10, collate_fn = DataUtils.collate_fn_padd_2,
        pin_memory=False)
    inputs,targets = next(iter(loader)) 
    assert inputs.shape == (10,1,998, 64)
    assert targets.size(0) == 10

#-----------------------------IEMOCAP------------------#
def test_birdsong_dataset_single():
    dataset = IEMOCAPDataset()
    mfcc = dataset.__getitem__(1784)
    assert mfcc[0].shape ==(745,64)
    


def test_birdsong_dataset_batch():
    dataset = IEMOCAPDataset()
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=10, collate_fn = DataUtils.collate_fn_padd_2,
        pin_memory=False)
    inputs,targets = next(iter(loader)) 
    assert len(dataset.labels_dict) ==10
    assert inputs.shape == (10,1,979, 64)
    assert targets.size(0) == 10

#-----------------------------Libri100------------------#

def test_libri100_dataset_single():
    dataset = Libri100()
    mfcc = dataset.__getitem__(1784)
    assert mfcc[0].shape ==(1534,64)
    assert mfcc[1] ==75

def test_libri100_dataset_batch():
    dataset = Libri100()
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=10,
        collate_fn = DataUtils.collate_fn_padd,
        pin_memory=False)
    inputs,targets = next(iter(loader)) 
    print(targets)
    assert inputs.shape == (10,1,1593, 64)
    assert targets.size(0) == 10

#---------------------------musical isnt-------------------------------#

def test_mi_dataset_single():
    dataset = MusicalInstrumentsDataset(type="valid")
    mfcc = dataset.__getitem__(1784)
    assert mfcc[0].shape ==(398,64)
    


def test_mi_dataset_batch():
    dataset = MusicalInstrumentsDataset(type="valid")
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=10, collate_fn = DataUtils.collate_fn_padd_2,
        pin_memory=False)
    inputs,targets = next(iter(loader)) 
    assert inputs.shape == (10,1,398, 64)
    assert targets.size(0) == 10


#--------------------- TuT Urban Sounds -------------------------
def test_tut_dataset_single():
    dataset = TutUrbanSounds(type="train")
    mfcc = dataset.__getitem__(1784)
    assert mfcc[0].shape ==(998,64)
    


def test_tut_dataset_batch():
    dataset = TutUrbanSounds(type="train")
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=10, collate_fn = DataUtils.collate_fn_padd_2,
        pin_memory=False)
    inputs,targets = next(iter(loader)) 
    assert inputs.shape == (10,1,998, 64)
    assert targets.size(0) == 10



#-----------------------------Speech Commands V2------------------#    

# def test_tf_speech_dataset_single():
#     dataset = TfSpeech()
#     mfcc = dataset.__getitem__(1784)
#     assert mfcc[0].shape == (1,98,64)

# def test_tf_speech_dataset_batch():
#     dataset = TfSpeech()
#     loader = torch.utils.data.DataLoader(dataset,
#         batch_size=10,
#         collate_fn = DataUtils.collate_fn_padd,
#         pin_memory=False)
#     inputs,targets = next(iter(loader)) 
#     assert inputs.shape == (10,1,98,64)
#     # assert targets.size(0) == 10    

#-----------------------------Speech Commands V2------------------#    

# def test_tf_speech_dataset_single():
#     dataset = TfSpeech()
#     mfcc = dataset.__getitem__(1784)
#     assert mfcc[0].shape == (1,98,64)

# def test_tf_speech_dataset_batch():
#     dataset = TfSpeech()
#     loader = torch.utils.data.DataLoader(dataset,
#         batch_size=10,
#         collate_fn = DataUtils.collate_fn_padd,
#         pin_memory=False)
#     inputs,targets = next(iter(loader)) 
#     assert inputs.shape == (10,1,98,64)
#     # assert targets.size(0) == 10    
