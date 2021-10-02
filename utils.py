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

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_downstream_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--down_stream_task', default="birdsong_freefield1010", type=str,
                        help='''down_stream task name one of 
                        birdsong_freefield1010 , birdsong_warblr ,
                        speech_commands_v1 , speech_commands_v2
                        libri_100 , musical_instruments , iemocap , tut_urban , voxceleb1 , musan
                        ''')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size ')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default = False, type=str2bool,
                        help='number of total epochs to run')
    parser.add_argument('--pretrain_path', default=None, type=str,
                        help='Path to Pretrain weights') 
    parser.add_argument('--freeze_effnet', default=True, type=str2bool,
                        help='Path to Pretrain weights')  
    parser.add_argument('--final_pooling_type', default='Avg', type=str,
                        help='valid final pooling types are Avg,Max') 
    parser.add_argument('--use_l2', default='l2', type=str,
                        help='datasetl2')                                                                                
    parser.add_argument('--load_only_efficientNet',default = True,type =str2bool)  
    parser.add_argument('--tag',default = "pretrain_big",type =str)                    
    return parser

#-----------------------------------------------------------------------------------------------#

def freeze_effnet(model):
    logger=logging.getLogger("__main__")
    logger.info("freezing effnet weights")
    for param in model.model_efficient.parameters():
        param.requires_grad = False

def load_pretrain(path,model,
                load_only_effnet=False,freeze_effnet=False):
    logger=logging.getLogger("__main__")
    logger.info("loading from checkpoint only weights : "+path)
    checkpoint = torch.load(path)
    if load_only_effnet :
        for key in checkpoint['state_dict'].copy():
            if not key.startswith('model_efficient'):
                del checkpoint['state_dict'][key]
    mod_missing_keys,mod_unexpected_keys   = model.load_state_dict(checkpoint['state_dict'],strict=False)
    logger.info("Model missing keys")
    logger.info(mod_missing_keys)
    logger.info("Model unexpected keys")
    logger.info(mod_unexpected_keys)
    if freeze_effnet : 
        logger.info("freezing effnet weights")
        for param in model.model_efficient.parameters():
            param.requires_grad = False
    logger.info("done loading")
    return model

def resume_from_checkpoint(path,model,optimizer):
    logger = logging.getLogger("__main__")
    logger.info("loading from checkpoint : "+path)
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']  
    logger.info("Task :: {}".format(checkpoint['down_stream_task']))
    mod_missing_keys,mod_unexpected_keys = model.load_state_dict(checkpoint['state_dict'],strict=False)  
    opt_missing_keys,opt_unexpected_keys = optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("Model missing keys",mod_missing_keys)
    logger.info("Model unexpected keys",mod_unexpected_keys)
    logger.info("Opt missing keys",opt_missing_keys)
    logger.info("Opt unexpected keys",opt_unexpected_keys)
    logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    return start_epoch , model, optimizer


def save_to_checkpoint(down_stream_task,dir,epoch,model,opt):
    torch.save({
            'down_stream_task': down_stream_task,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : opt.state_dict()
            },
            os.path.join('.',dir,'models', 'checkpoint_' + str(epoch) + "_" + '.pth.tar')
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
