import argparse
import os
import pickle
import time

# import faiss
import numpy as np
# from sklearn.metrics.cluster import normalized_mutual_info_score

#import clustering
#import models
#from util import AverageMeter, Logger, UnifLabelSampler

from os.path import join as path_join
import json
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
import torch
from torch import nn
# import librosa
# import tensorflow as tf
from efficientnet_pytorch import EfficientNet
# from scipy.sparse import csr_matrix, find
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

import logging
logging.basicConfig(filename='train.log', filemode='w')
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------------#
import argparse

from datasets.birdsong_dataset import BirdSongDataset
from datasets.tf_speech import TfSpeech

def get_data_set(name):
    if name == "tf_speech" :
        return TfSpeech()
    elif name == "bird_song":
        return BirdSongDataset()    
    else :
        raise NotImplementedError

def get_downstream_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--down_stream_task', default="tf_speech", type=str,
                        help='down_stream task name')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size ')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--no_of_classes', default=2, type=int, metavar='N',
                        help='number of classes')
    return parser

#-----------------------------------------------------------------------------------------------#

def resume_from_checkpoint(path,model,optimizer):
    logger.info("loading from checkpoint : "+path)
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    # remove top_layer parameters from checkpoint
    for key in checkpoint['state_dict'].copy():
        if 'top_layer' in key:
            del checkpoint['state_dict'][key]
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    return start_epoch , model, optimizer


def save_to_checkpoint(down_stream_task,epoch,model,opt):
    torch.save({
            'down_stream_task': down_stream_task,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : opt.state_dict()
            },
            os.path.join('.', 'checkpoint_' + str(epoch) + "_" + '.pth.tar')
    )

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#

class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if isinstance(val, (torch.Tensor)):
            val = val.numpy()
            self.val = val
            self.sum += np.sum(val) 
            self.count += np.size(val)
        self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)
