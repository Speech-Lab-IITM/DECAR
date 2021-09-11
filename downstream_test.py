import argparse
import os
import pickle
import time
import numpy as np
from utils import AverageMeter,UnifLabelSampler, get_downstream_parser 

import torch
from torch import nn

from datasets.birdsong_dataset import BirdSongDataset
from datasets.tf_speech import TfSpeech
from datasets.tf_speech import TfSpeechTest
from datasets.data_utils import DataUtils
from efficientnet.model import DeepCluster_ICASSP
from utils import Metric 
from utils import resume_from_checkpoint
from utils import save_to_checkpoint
from tqdm import tqdm
import pandas as pd

import logging

def get_logger(file_name):
    logging.basicConfig(filename=file_name+'train.log', filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger


def get_dataset(downstream_task_name,mode="train"):
    if(mode=="train"):
        if downstream_task_name == "bird_song":
            return BirdSongDataset()
        elif downstream_task_name == "tf_speech":
            return TfSpeech()
        else:
            raise NotImplementedError
    elif(mode=="test"):
        if downstream_task_name == "bird_song":
            raise NotImplementedError
        elif downstream_task_name == "tf_speech":
            return TfSpeechTest()
        else:
            raise NotImplementedError

def set_seed(seed = 31):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def move_to_gpu(*args):
    if torch.cuda.is_available():
        for item in args:
            item.cuda()


def test(args):    
    logger = get_logger(args.down_stream_task)
    dataset = get_dataset(args.down_stream_task,mode="test")
    df_pred = pd.DataFrame(columns = [ 'fname','label'])
    model = DeepCluster_ICASSP(no_of_classes=dataset.no_of_classes)
    move_to_gpu(model)

    
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                collate_fn = DataUtils.collate_fn_padd,
                                                pin_memory=True)                                            

    for input_tensor, Ids in tqdm(test_loader):
        # print(Ids)
        # print(input_tensor.shape)
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input_tensor.cuda())
        else:
            input_var = torch.autograd.Variable(input_tensor)

        output = model(input_var)
        preds = torch.argmax(output,dim=1)
        for i in range(output.shape[0]):
            filename = Ids[i]
            label = dataset.get_label(preds[i].cpu().numpy())
            df_pred=df_pred.append( {'fname' : filename, 'label' : label} , ignore_index = True)
        #break

    df_pred.to_csv('/speech/sandesh/icassp/tf_speech/test/final_preds.csv',index=False)






def main():
    set_seed()
    parser = get_downstream_parser()
    args = parser.parse_args()
    test(args)

if __name__== "__main__":
    main()

