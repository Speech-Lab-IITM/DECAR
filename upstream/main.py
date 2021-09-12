import argparse
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch.backends.cudnn as cudnn
#import clustering
#import models
from utils import AverageMeter,UnifLabelSampler
#, Logger, UnifLabelSampler

from os.path import join as path_join
import json
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample
import torch
from torch import nn
import librosa
import tensorflow as tf
from efficientnet_pytorch import EfficientNet
from scipy.sparse import csr_matrix, find
import torch.utils.data as data


from utils import extract_log_mel_spectrogram
from utils import compute_features
from utils import run_kmeans
from utils import preprocess_features
from utils import Kmeans
from specaugment import specaug


#------------------------------------------------------------------------------------------------------------------------------------------------------------------#
AUDIO_SR = 16000

list_of_files_directory = os.listdir("/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/")

tf.config.set_visible_devices([], 'GPU')

def collate_fn_padd_1(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## padd
    batch = [torch.Tensor(t) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch,batch_first = True)
    #batch = batch.reshape()
    batch = batch.unsqueeze(1)

    return batch


def collate_fn_padd_2(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## padd
    
    batch_x = [torch.Tensor(t) for t,y in batch]
    batch_y = [y for t,y in batch]
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first = True)
    batch_x = batch_x.unsqueeze(1)
    batch_y = torch.Tensor(batch_y).type(torch.LongTensor)

    return batch_x,batch_y

#------------------------------------------------------------------------------------------------------------------------------------------------------------------#
class DeepCluster(Dataset):
    def __init__(self, data_dir, data_dir_list):
        self.datafolder = data_dir
        self.audio_files_list = data_dir_list
        
    def __getitem__(self, idx):
        audio_file = os.path.join(self.datafolder, self.audio_files_list[idx])
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        log_mel_spec = extract_log_mel_spectrogram(wave)
        log_mel_spec = log_mel_spec.numpy().tolist()       
        return log_mel_spec

    def __len__(self):
        return len(self.audio_files_list)

class DeepCluster_Reassigned(Dataset):
    def __init__(self,data_dir, audio_file_list,label_list,audio_indexes):
        self.data_directory = data_dir
        self.audio_files = audio_file_list
        self.audio_labels = label_list
        self.audio_indexes = audio_indexes
        self.dataset = self.make_dataset()
        #print(self.dataset)
        #print(self.audio_files_list)
        
    def make_dataset(self):
        label_to_idx = {label: idx for idx, label in enumerate(set(self.audio_labels))}
        audiopath_w_labels = []
        for i, index in enumerate(self.audio_indexes):
            path = self.audio_files[index]
            pseudolabel = label_to_idx[self.audio_labels[index]] #could have been pseudolabels, bekar confusion change later
            audiopath_w_labels.append((path,pseudolabel))
            
        return audiopath_w_labels
            
    def __getitem__(self, idx):
        audio_file,label = self.dataset[idx]
        audio_file= os.path.join(self.data_directory,audio_file)
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        log_mel_spec = extract_log_mel_spectrogram(wave)
        log_mel_spec = torch.tensor(log_mel_spec.numpy().tolist())
        log_mel_spec = specaug(log_mel_spec.clone().detach().requires_grad_(True))        
        return log_mel_spec,label

    def __len__(self):
        return len(self.audio_files)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------#
class DeepCluster_ICASSP(nn.Module):
    def __init__(self):
        super(DeepCluster_ICASSP, self).__init__()
        self.model_efficient = EfficientNet.from_name('efficientnet-b0',include_top = False, in_channels = 1,image_size = None)
        #self.classifier = nn.Sequential(nn.Dropout(0.5),nn.Linear(1280, 512),nn.ReLU(),nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.5),nn.Linear(1280, 1280),nn.ReLU())
        #self.top_layer = nn.Linear(512, 256)
        self.top_layer = nn.Linear(1280, 256)
    def forward(self,batch):
        x = self.model_efficient(batch)
        x = x.flatten(start_dim=1) #1280 (already swished)
        x = self.classifier(x)
        
        if self.top_layer:
            x = self.top_layer(x)
        
        return x
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def main():
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    np.random.seed(31)

    final_model = DeepCluster_ICASSP()

    fd = int(final_model.top_layer.weight.size()[1])
    final_model.top_layer = None
    final_model.model_efficient = torch.nn.DataParallel(final_model.model_efficient)

    #final_model.features = torch.nn.DataParallel(final_model.features)
    final_model.cuda()
    cudnn.benchmark = True


    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, final_model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**-5,
    )


    criterion = nn.CrossEntropyLoss().cuda()

    #Resume from checkpoint
    print("loading checkpoint")
    checkpoint = torch.load("/speech/srayan/icassp/training/best_loss.pth.tar")
    start_epoch = checkpoint['epoch']
    # remove top_layer parameters from checkpoint
    for key in checkpoint['state_dict'].copy():
        if 'top_layer' in key:
            del checkpoint['state_dict'][key]
    final_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))



    pretrain_dataset = DeepCluster("/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/",list_of_files_directory)

    train_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=16,collate_fn = collate_fn_padd_1,pin_memory=True,num_workers = 2)

    best_loss = float("inf")

    #for epoch in range(0,200):
    for epoch in range(start_epoch,200): #when using checkpoint

        final_model.top_layer = None

        final_model.classifier = nn.Sequential(*list(final_model.classifier.children())[:-1])

        features = compute_features(train_loader, final_model, len(list_of_files_directory))

        print("Entering Clustering")

        deepcluster = Kmeans(256)
        clustering_loss = deepcluster.cluster(features, verbose=True)

        print("Done Clustering")

        mlp = list(final_model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        final_model.classifier = nn.Sequential(*mlp)
        final_model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        final_model.top_layer.weight.data.normal_(0, 0.01)
        final_model.top_layer.bias.data.zero_()
        final_model.top_layer.cuda()


        print("Starting To make Reassigned Dataset")

        pseudolabels = []
        image_indexes = []
        for cluster, images in enumerate(deepcluster.images_lists):
            image_indexes.extend(images)
            pseudolabels.extend([cluster] * len(images))

        indexes_sorted = np.argsort(image_indexes)  
        pseudolabels = np.asarray(pseudolabels)[indexes_sorted]
    
        #label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}

        dataset_w_labels = DeepCluster_Reassigned("/speech/srayan/icassp/kaggle_data/audioset_train/train_wav/",list_of_files_directory,pseudolabels,indexes_sorted)

        sampler = UnifLabelSampler(int(len(list_of_files_directory)),deepcluster.images_lists)

        train_loader_reassigned = torch.utils.data.DataLoader(dataset_w_labels,batch_size=64,collate_fn = collate_fn_padd_2,sampler=sampler,pin_memory=True,num_workers = 2)

        print("Starting To Train")

        loss = train(train_loader_reassigned, final_model, criterion, optimizer, epoch)

        torch.save({'epoch': epoch + 1,
                    'state_dict': final_model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join('/speech/srayan/icassp/training/checkpoints_deepcluster/', 'checkpoint_' + str(epoch + 1) + "_" + '.pth.tar'))

        if loss < best_loss:
            torch.save({'epoch': epoch + 1,
                    'state_dict': final_model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join('/speech/srayan/icassp/training/', 'best_loss.pth.tar'))
            best_loss = loss


def train(loader, model, crit, opt, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    model.train()

    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=0.05,
        weight_decay=10**-5,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        #print("One iteration done")
        data_time.update(time.time() - end)

        n = len(loader) * epoch + i

        if n % 5000 == 0:
            print('Saving Checkpoint')
            path = os.path.join(
                "/speech/srayan/icassp/training/",
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

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return loss



if __name__== "__main__":
    main()


















