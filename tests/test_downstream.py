import torch
import pytest
import numpy as np

from efficientnet.model import DeepCluster_ICASSP
from datasets.birdsong_dataset import BirdSongDataset
from utils import Metric

# test using :  python -m pytest tests/

def test_model_ouputs():
    deep_model = DeepCluster_ICASSP(2)
    forward = deep_model(torch.randn(10,1,200,40))
    assert forward.shape == (10,2)



def test_birdsong_dataset_single():
    dataset = BirdSongDataset()
    mfcc = dataset.__getitem__(1784)
    assert mfcc[0].shape ==(1,2206,30)
    


def test_birdsong_dataset_batch():
    dataset = BirdSongDataset()
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=10,# collate_fn = collate_fn_padd_1,
        pin_memory=False)
    inputs,targets = next(iter(loader)) 
    assert inputs.shape == (10,1,2206,30)
    assert targets.size(0) == 10

def test_metric():
    metric = Metric()
    metric.update([1,1.2])
    assert metric.avg == 1.1