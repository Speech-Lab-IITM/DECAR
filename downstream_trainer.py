import argparse
import os
import pickle
import time
import numpy as np
from utils import AverageMeter,UnifLabelSampler, get_downstream_parser 

from os.path import join as path_join
import json
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
import torch
from torch import nn

import torch.utils.data as data
from tqdm import tqdm

from datasets.birdsong_dataset import BirdSongDataset
from datasets.tf_speech import TfSpeech
from datasets.data_utils import DataUtils
from efficientnet.model import DeepCluster_ICASSP
from utils import Metric 
from utils import resume_from_checkpoint
from utils import save_to_checkpoint

import logging

def get_logger(file_name):
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(os.path.join("/speech/sandesh/icassp/deep_cluster",
                                            "logs",file_name+'train.log'))
    f_handler.setLevel(logging.DEBUG)
    # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_dataset(downstream_task_name):
    if downstream_task_name == "bird_song":
        return BirdSongDataset()
    elif downstream_task_name == "tf_speech":
        return TfSpeech()
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


def train(args):    
    start_epoch=1
    logger = get_logger(args.down_stream_task)
    # logger.info("Starting To Train")
    # exit()
    dataset = get_dataset(args.down_stream_task)

    # -----------model criterion optimizer ---------------- #
    model = DeepCluster_ICASSP(no_of_classes=dataset.no_of_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**-5,
    )

    move_to_gpu(model,criterion)

    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                collate_fn = DataUtils.collate_fn_padd,
                                                pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=args.batch_size,
                                                collate_fn = DataUtils.collate_fn_padd,
                                                pin_memory=True)                                            

    best_loss = float("inf")

    train_stats = eval(train_loader, model, criterion)
    logger.info("epoch :{} Train loss: {} Train accuracy: {} Valid loss: {} Valid accuracy: {}".format(
            -1 , train_stats["loss"].avg.numpy() ,train_stats["accuracy"].avg,
                0,0
        ) )
    logger.info("Starting To Train")
    for epoch in range(start_epoch,args.epochs):
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, epoch,logger)
        train_loss = train_stats["loss"]
        save_to_checkpoint(args.down_stream_task,
                            epoch,model,optimizer)
        eval_stats = eval(valid_loader,model,criterion)

        logger.info("epoch :{} Train loss: {} Train accuracy: {} Valid loss: {} Valid accuracy: {}".format(
            epoch , train_stats["loss"].avg.numpy() ,train_stats["accuracy"].avg,
                eval_stats["loss"].avg.numpy() , eval_stats["accuracy"].avg
        ) )
        print("epoch :{} Train loss: {} Train accuracy: {} Valid loss: {} Valid accuracy: {}".format(
            epoch , train_stats["loss"].avg.numpy() ,train_stats["accuracy"].avg,
                eval_stats["loss"].avg.numpy() , eval_stats["accuracy"].avg
        ) )
        # logger.info("Train loss: "+str(eval_metrics["loss"].avg.numpy()))
        # logger.info("Train accuracy: "+str(eval_metrics["accuracy"].avg))

        # if train_loss < best_loss:
        #     # TODO :: point best to best epoch model 
        #     best_loss = train_loss


def train_one_epoch(loader, model, crit, opt, epoch,logger):
    '''
    Train one Epoch
    '''
    logger.debug("epoch:"+str(epoch) +" Started")
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    model.train()
    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # n = len(loader) * epoch + i
        # if n % 5000 == 0:
        #     print('Saving Checkpoint')
        #     # path = os.path.join(
        #     #     "/speech/srayan/icassp/training/",
        #     #     'checkpoints',
        #     #     'checkpoint_' + str(n / 5000) + '.pth.tar',
        #     # )

        #     # torch.save({
        #     #     'epoch': epoch + 1,
        #     #     'state_dict': model.state_dict(),
        #     #     'optimizer' : opt.state_dict()
        #     # }, path)

        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            target = target.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target)
        else:
            input_var = torch.autograd.Variable(input_tensor)
            target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)
        
        # record loss
        losses.update(loss.data, input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))
    
    
    #--------------epoch evaultion-----------------# 

    eval_metrics =eval(loader,model,crit)
    logger.debug("epoch-"+str(epoch) +" ended")
    return eval_metrics

def eval(loader, model,crit):
    model.eval()
    losses = AverageMeter()
    accuracy = Metric()
    for i, (input_tensor, targets) in enumerate(loader):
        if torch.cuda.is_available():
            input_tensor =input_tensor.cuda()
            targets = targets.cuda()
        outputs = model(input_tensor)
        preds = torch.argmax(outputs,dim=1)==targets
        accuracy.update(preds.cpu())
        loss = crit(outputs, targets)
        losses.update(loss.cpu().data, input_tensor.size(0))

    metrics_dict={"accuracy" : accuracy , "loss" : losses}    
    return metrics_dict


def main():
    set_seed()
    parser = get_downstream_parser()
    args = parser.parse_args()
    train(args)

if __name__== "__main__":
    main()

