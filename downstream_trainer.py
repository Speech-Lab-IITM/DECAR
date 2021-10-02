import argparse
import os
import pickle
import time
import numpy as np
from utils import AverageMeter,UnifLabelSampler, create_dir, get_downstream_parser , load_pretrain ,freeze_effnet

from os.path import join as path_join
import json
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
import torch
from torch import nn

import torch.utils.data as data
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets.dataset import get_dataset
from datasets.data_utils import DataUtils
from efficientnet.model import DeepCluster_downstream
from utils import Metric 
from utils import resume_from_checkpoint
from utils import save_to_checkpoint
import random
import logging
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_logger(args):
    create_dir(args.exp_root)
    create_dir(os.path.join(args.exp_root,'models'))
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(os.path.join(args.exp_root,'train.log'))
    f_handler.setLevel(logging.INFO)
    # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)
    return logger

def log_args(args):
    logger = logging.getLogger(__name__)
    logger.info("Downstream Task {}".format(args.down_stream_task))
    logger.info("Resume {}, load only efficient net {}, from path {} ".format(args.resume,
                args.load_only_efficientNet,args.pretrain_path))
    logger.info("BS {}".format(args.batch_size))  
    logger.info("complete args %r",args)         

def set_seed(seed = 31):
    random.seed(seed)
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
    args.exp_root = os.path.join('.','exp',args.down_stream_task,args.final_pooling_type,args.tag)
    logger = get_logger(args)
    log_args(args)
    train_dataset,valid_dataset = get_dataset(args.down_stream_task)

    # -----------model criterion optimizer ---------------- #
    model = DeepCluster_downstream(no_of_classes=train_dataset.no_of_classes,final_pooling_type=args.final_pooling_type)
    model.model_efficient = torch.nn.DataParallel(model.model_efficient)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=0.001,# momentum=0.9,weight_decay=10**-5,
    )
    logger.info(str(model))
    if args.resume:
        resume_from_checkpoint(args.pretrain_path,model,optimizer)
    elif args.pretrain_path:
        load_pretrain(args.pretrain_path,model,args.load_only_efficientNet,args.freeze_effnet)
    elif args.freeze_effnet:
        logger.info("Random Weights init")
        freeze_effnet(model)
    else:
        logger.info("Random Weights init")
        pass

    move_to_gpu(model,criterion)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                collate_fn = DataUtils.collate_fn_padd_2,
                                                pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=args.batch_size,
                                                collate_fn = DataUtils.collate_fn_padd_2,
                                                pin_memory=True)                                            

    # train_stats = eval(train_loader, model, criterion)
    # eval_stats = eval(valid_loader,model,criterion)
    # logger.info("epoch :{} Train loss: {} Train accuracy: {} Valid loss: {} Valid accuracy: {}".format(
    #         -1 , train_stats["loss"].avg.numpy() ,train_stats["accuracy"].avg,
    #             eval_stats["loss"].avg.numpy() , eval_stats["accuracy"].avg
    #     ) )
    logger.info("Starting To Train")
    train_accuracy = []
    train_losses=[]
    valid_accuracy = []
    valid_losses=[]
    for epoch in range(start_epoch,args.epochs+1):
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        train_loss = train_stats["loss"]
        save_to_checkpoint(args.down_stream_task,args.exp_root,
                            epoch,model,optimizer)
        eval_stats = eval(valid_loader,model,criterion)
        train_losses.append(train_stats["loss"].avg.numpy())
        train_accuracy.append(train_stats["accuracy"].avg)
        valid_losses.append(eval_stats["loss"].avg.numpy())
        valid_accuracy.append(eval_stats["accuracy"].avg)

        logger.info("epoch :{} Train loss: {} Train accuracy: {} Valid loss: {} Valid accuracy: {}".format(
            epoch , train_stats["loss"].avg.numpy() ,train_stats["accuracy"].avg,
                eval_stats["loss"].avg.numpy() , eval_stats["accuracy"].avg
        ) )
    logger.info("max train accuracy : {}".format(max(train_accuracy)))
    logger.info("max valid accuracy : {}".format(max(valid_accuracy)))
    plt.plot(range(1,len(train_accuracy)+1), train_accuracy, label = "train accuracy",marker = 'x')
    plt.plot(range(1,len(valid_accuracy)+1), valid_accuracy, label = "valid accuracy",marker = 'x')
    plt.legend()
    plt.savefig(os.path.join(args.exp_root,'accuracy.png'))

def train_one_epoch(loader, model, crit, opt, epoch):
    '''
    Train one Epoch
    '''
    logger = logging.getLogger(__name__)
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
    print(args)
    train(args)

if __name__== "__main__":
    main()

