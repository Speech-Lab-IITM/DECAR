import argparse
import os
import pickle
import time
import numpy as np
from utils import AverageMeter,UnifLabelSampler 

from os.path import join as path_join
import json
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
import torch
from torch import nn

import torch.utils.data as data

from datasets.birdsong_dataset import BirdSongDataset
from efficientnet.model import DeepCluster_ICASSP
from utils import Metric

import logging
logging.basicConfig(filename='train.log', filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def set_seed(seed = 31):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def move_to_gpu(*args):
    if torch.cuda.is_available():
        for item in args:
            item.cuda()

def resume_from_checkpoint(path):
    logger.info("loading from checkpoint : "+path)
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    # remove top_layer parameters from checkpoint
    for key in checkpoint['state_dict'].copy():
        if 'top_layer' in key:
            del checkpoint['state_dict'][key]
    final_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

def save_to_checkpoint(epoch,model,opt):
    torch.save({'epoch': epoch ,
                'state_dict': final_model.state_dict(),
                'optimizer' : optimizer.state_dict()},
                os.path.join('.', 'checkpoint_' + str(epoch) + "_" + '.pth.tar'))

def main():
    set_seed()
    start_epoch=1
    final_model = DeepCluster_ICASSP(no_of_classes=2)

    # fd = int(final_model.top_layer.weight.size()[1])
    # final_model.top_layer = None
    # final_model.model_efficient = torch.nn.DataParallel(final_model.model_efficient)
    # final_model.features = torch.nn.DataParallel(final_model.features)
    # final_model.cuda()
    # cudnn.benchmark = True


    #----------define optimiser-------------#
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, final_model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**-5,
    )
    criterion = nn.CrossEntropyLoss()

    move_to_gpu(final_model,criterion)



    train_dataset = BirdSongDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32,# collate_fn = collate_fn_padd_1,
                                                pin_memory=True)

    best_loss = float("inf")
    # eval_metrics =eval(train_loader,final_model,criterion)
    # logger.info("epoch-"+str(start_epoch) +" metrics:")
    # logger.info("Train loss: "+str(eval_metrics["loss"].avg.numpy()))
    # logger.info("Train accuracy: "+str(eval_metrics["accuracy"].avg))
    # logger.info("epoch-"+str(start_epoch) +" ended")
    # exit()
    logger.info("Starting To Train")
    for epoch in range(start_epoch,200):
        loss = train_one_epoch(train_loader, final_model, criterion, optimizer, epoch)
        # save_to_checkpoint(epoch,final_model,optimiser)

        # if loss < best_loss:
        #     torch.save({'epoch': epoch + 1,  # ???????? epoch +1
        #             'state_dict': final_model.state_dict(),
        #             'optimizer' : optimizer.state_dict()},
        #             os.path.join('.', 'best_loss.pth.tar'))
        #     best_loss = loss


def train_one_epoch(loader, model, crit, opt, epoch):
    '''
    Train one Epoch
    '''
    logger.info("epoch:"+str(epoch) +" Started")
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    model.train()
    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        n = len(loader) * epoch + i

        if n % 5000 == 0:
            print('Saving Checkpoint')
            # path = os.path.join(
            #     "/speech/srayan/icassp/training/",
            #     'checkpoints',
            #     'checkpoint_' + str(n / 5000) + '.pth.tar',
            # )

            # torch.save({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'optimizer' : opt.state_dict()
            # }, path)

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
    logger.info("epoch-"+str(epoch) +" metrics:")
    logger.info("Train loss: "+str(eval_metrics["loss"].avg.numpy()))
    logger.info("Train accuracy: "+str(eval_metrics["accuracy"].avg))
    logger.info("epoch-"+str(epoch) +" ended")
    return loss

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


if __name__== "__main__":
    main()

