import argparse
import os
import pickle
import time
import json
import torch
import librosa
import tensorflow as tf
import numpy as np

from os.path import join as path_join
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


def extract_log_mel_spectrogram(waveform,
                                sample_rate=16000,
                                frame_length=400,
                                frame_step=160,
                                fft_length=1024,
                                n_mels=64,
                                fmin=60.0,
                                fmax=7800.0):
  """Extract frames of log mel spectrogram from a raw waveform."""

  stfts = tf.signal.stft(
      waveform,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length)
  spectrograms = tf.abs(stfts)

  num_spectrogram_bins = stfts.shape[-1]
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  mel_spectrograms = tf.clip_by_value(
      mel_spectrograms,
      clip_value_min=1e-5,
      clip_value_max=1e8)

  log_mel_spectrograms = tf.math.log(mel_spectrograms)

  return log_mel_spectrograms

def compute_features(args, dataloader, model, N): #N is total dataset size
    batch = args.batch_size
    verbose = True
    if verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
                # special treatment for final batch
                features[i * batch:] = aux

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if verbose and (i % 50) == 0:
                print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
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


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

def get_upstream_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size ')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume',action='store_true',
                        help='number of total epochs to run')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Path to Pretrain weights')  
    parser.add_argument('--save_dir', default=None, type=str,
                        help='Path to Save Checkpoints and Logs')
    parser.add_argument('--num_clusters', default=None, type=int,
                        help='Number of clusters in case of K-Means')
    parser.add_argument('--cluster_algo', default='kmeans', type=str,
                        help='Choice of clustering algorithm')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers')
    parser.add_argument('--input', type=str, required = True, 
                        help='Input CSV with all file paths under column named files')
  
    return parser
