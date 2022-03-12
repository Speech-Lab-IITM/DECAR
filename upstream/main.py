import argparse
import os
import pickle
import time
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics.cluster import normalized_mutual_info_score
from os.path import join as path_join
import json
import torch
import tensorflow as tf
import logging
from torch import nn


from utils import extract_log_mel_spectrogram, compute_features, get_upstream_parser, AverageMeter, UnifLabelSampler, Logger
from clustering import run_kmeans, Kmeans, PIC, rearrange_clusters
from specaugment import specaug
from datasets import collate_fn_padd_1, collate_fn_padd_2, DeepCluster, DeepCluster_Reassigned
from models import AudioNTT2020
from specaugment import specaug
from augmentations import MixupBYOLA, RandomResizeCrop, RunningNorm
from models import AudioNTT2020
import pandas as pd

AUDIO_SR = 16000


logging.basicConfig(filename='decar.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

class AugmentationModule:
    """BYOL-A augmentation module example, the same parameter with the paper."""

    def __init__(self, args, size, epoch_samples, log_mixup_exp=True, mixup_ratio=0.4):
        self.train_transform = nn.Sequential(
            MixupBYOLA(ratio=mixup_ratio, log_mixup_exp=log_mixup_exp),
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)),
        )
        self.pre_norm = RunningNorm(epoch_samples=epoch_samples)
        print('Augmentations:', self.train_transform)
        self.norm_status = args.use_norm
    def __call__(self, x):
        if self.norm_status == "byol":
            x = self.pre_norm(x)
        return self.train_transform(x)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(gpu, args):
    
    args.rank += gpu
    
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    np.random.seed(31)

    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    list_of_files_directory = pd.read_csv('/speech/ashish/small.csv')
    list_of_files_directory = list(list_of_files_directory["files"])

    #final_model = DeepCluster_ICASSP()
    #final_model.model_efficient = torch.nn.DataParallel(final_model.model_efficient)
    print('model defination')
    final_model = AudioNTT2020(args, args.feat_dim, n_mels=64, d=2048).cuda(gpu)
    print(final_model.classifier)
    fd = int(final_model.top_layer.weight.size()[1])
    final_model.top_layer = None
    final_model = nn.SyncBatchNorm.convert_sync_batchnorm(final_model)

    final_model = nn.parallel.DistributedDataParallel(final_model, device_ids=[gpu],find_unused_parameters=True)
    print('initilization of the model completed')
    print(final_model)
    
    
    #final_model.cuda()
    logger.info(final_model)
    cudnn.benchmark = True


    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, final_model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**-5,
    )

    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0

    #Resume from checkpoint
    if args.resume:
        logger.info("loading checkpoint")
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint['epoch']
        # remove top_layer parameters from checkpoint
        for key in checkpoint['state_dict'].copy():
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
        final_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    cluster_log = Logger(os.path.join(args.save_dir, 'clusters'))

    pretrain_dataset = DeepCluster(list_of_files_directory,len(list_of_files_directory),args)  #without augmentation 

    train_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=args.batch_size)

    best_loss = float("inf")
    tfms = AugmentationModule(args, (64, 96), 2 * len(list_of_files_directory))

    for epoch in range(start_epoch,args.epochs): #when using checkpoint

        final_model.module.top_layer = None
        #during clustering we do not do ReLU
        final_model.module.classifier = nn.Sequential(*list(final_model.module.classifier.children())[:-1])

        features = compute_features(args, train_loader, final_model, len(list_of_files_directory))

        logger.info("Entering Clustering")

        if args.cluster_algo == "kmeans":
            deepcluster = Kmeans(args.num_clusters)
        else:
            deepcluster = PIC(args.num_clusters) #random value not used
        
        clustering_loss = deepcluster.cluster(features, verbose=True)

        logger.info("Done Clustering")

        mlp = list(final_model.module.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        final_model.module.classifier = nn.Sequential(*mlp)
        final_model.module.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        final_model.module.top_layer.weight.data.normal_(0, 0.01)
        final_model.module.top_layer.bias.data.zero_()
        final_model.module.top_layer.cuda()

        logger.info("Starting To make Reassigned Dataset")

        pseudolabels = []
        image_indexes = []
        for cluster, images in enumerate(deepcluster.images_lists):
            image_indexes.extend(images)
            pseudolabels.extend([cluster] * len(images))

        indexes_sorted = np.argsort(image_indexes)  
        pseudolabels = np.asarray(pseudolabels)[indexes_sorted]

        #Count number of clusters having assignments
        count = 0
        for l in deepcluster.images_lists:
            if len(l) > 0:
                count += 1

        logger.info("Number of clusters having assignments is = " + str(count))
    
        dataset_w_labels = DeepCluster_Reassigned(args,list_of_files_directory,pseudolabels,indexes_sorted,tfms)

        sampler = UnifLabelSampler(int(len(list_of_files_directory)),deepcluster.images_lists)

        train_loader_reassigned = torch.utils.data.DataLoader(dataset_w_labels,batch_size=args.batch_size,collate_fn = collate_fn_padd_2,sampler=sampler,pin_memory=True,num_workers=args.num_workers)

        logger.info("Starting To Train")

        loss = train(args, train_loader_reassigned, final_model, criterion, optimizer, epoch)

        logger.info("Logging and saving checkpoints")

        logger.info('###### Epoch [{0}] ###### \n'
                  'Clustering loss: {1:.3f} \n'
                  'ConvNet loss: {2:.3f}'
                  .format(epoch, clustering_loss, loss))
        try:
            nmi = normalized_mutual_info_score(
                    pseudolabels,
                    rearrange_clusters(cluster_log.data[-1])
                )
            logger.info('NMI against previous assignment: {0:.3f}'.format(nmi))

        except IndexError:
            pass
        logger.info('####################### \n')

        #Save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'state_dict': final_model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join(args.save_dir, 'checkpoints_deepcluster', 'checkpoint_' + str(epoch + 1) + "_" + '.pth.tar'))

        #Save best checkpoint
        if epoch > 0:
            if loss < best_loss:
                torch.save({'epoch': epoch + 1,
                    'state_dict': final_model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join(args.save_dir, 'best_loss.pth.tar'))
                best_loss = loss
        
        cluster_log.log(deepcluster.images_lists)


def train(args, loader, model, crit, opt, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    model.train()

    optimizer_tl = torch.optim.SGD(
        model.module.top_layer.parameters(),
        lr=0.05,
        weight_decay=10**-5,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        n = len(loader) * epoch + i

        if n % 5000 == 0:
            logger.info('Saving Checkpoint')
            path = os.path.join(
                args.save_dir,
                'checkpoints',
                'checkpoint_' + str(n / 5000) + '.pth.tar',
            )

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)


        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)


        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.data, input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        batch_time.update(time.time() - end)
        end = time.time()

        logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))
        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))                  

    return losses.avg


if __name__== "__main__":
    parser = get_upstream_parser()
    args = parser.parse_args()

    create_dir(os.path.join(args.save_dir,'checkpoints'))
    create_dir(os.path.join(args.save_dir,'checkpoints_deepcluster'))

    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = 1

    torch.multiprocessing.spawn(main, (args,), 1)


















