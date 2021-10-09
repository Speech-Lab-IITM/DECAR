import logging
import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn

from datasets.data_utils import DataUtils
from datasets.dataset import get_dataset
from efficientnet.model import DeepCluster_downstream
from utils import (AverageMeter, Metric, create_dir, freeze_effnet,
                   get_downstream_parser, load_pretrain,
                   resume_from_checkpoint, save_to_checkpoint,move_to_gpu,set_seed)

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


def train(args):    
    start_epoch=1
    args.exp_root = os.path.join('.','exp',args.down_stream_task,args.final_pooling_type,args.tag)
    logger = get_logger(args)
    log_args(args)
    train_dataset,test_dataset = get_dataset(args.down_stream_task)

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
    else:
        logger.info("Random Weights init")
        pass

    if args.freeze_effnet:
        freeze_effnet(model)

    move_to_gpu(model,criterion)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                collate_fn = DataUtils.collate_fn_padd_2,
                                                pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_size,
                                                collate_fn = DataUtils.collate_fn_padd_2,
                                                pin_memory=True)                                            

    logger.info("Starting To Train")
    train_accuracy = []
    train_losses=[]
    test_accuracy = []
    test_losses=[]
    for epoch in range(start_epoch,args.epochs+1):
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        save_to_checkpoint(args.down_stream_task,args.exp_root,
                            epoch,model,optimizer)
        eval_stats = eval(test_loader,model,criterion)
        train_losses.append(train_stats["loss"].avg.numpy())
        train_accuracy.append(train_stats["accuracy"].avg)
        test_losses.append(eval_stats["loss"].avg.numpy())
        test_accuracy.append(eval_stats["accuracy"].avg)

        logger.info("epoch :{} Train loss: {} Train accuracy: {} Valid loss: {} Valid accuracy: {}".format(
            epoch , train_stats["loss"].avg.numpy() ,train_stats["accuracy"].avg,
                eval_stats["loss"].avg.numpy() , eval_stats["accuracy"].avg
        ) )
    logger.info("max train accuracy : {}".format(max(train_accuracy)))
    logger.info("max valid accuracy : {}".format(max(test_accuracy)))
    plt.plot(range(1,len(train_accuracy)+1), train_accuracy, label = "train accuracy",marker = 'x')
    plt.plot(range(1,len(test_accuracy)+1), test_accuracy, label = "valid accuracy",marker = 'x')
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

    model.train()
    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            target = target.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target)
        else:
            input_var = torch.autograd.Variable(input_tensor)
            target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)
        
        losses.update(loss.data, input_tensor.size(0))
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

