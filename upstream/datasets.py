import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset
from augmentations import MixupBYOLA, RandomResizeCrop, RunningNorm
from specaugment import specaug
from utils import extract_log_mel_spectrogram, extract_window, extract_log_mel_spectrogram_torch, extract_window_torch, MelSpectrogramLibrosa
import librosa
AUDIO_SR = 16000
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
    #batch_x = batch_x.unsqueeze(1)
    batch_y = torch.Tensor(batch_y).type(torch.LongTensor)

    return batch_x,batch_y


class DeepCluster(Dataset):

    def __init__(self, data_dir_list,epoch_samples,args):
        self.audio_files_list = data_dir_list
        self.norm_status = args.use_norm
        self.pre_norm = RunningNorm(epoch_samples=epoch_samples)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.length = args.length_wave

    def __getitem__(self, idx):
        audio_file = self.audio_files_list[idx]
        #wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        #log_mel_spec = extract_log_mel_spectrogram(wave)
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        wave = torch.tensor(wave)
        
        waveform = extract_window_torch(self.length, wave) #extract a window
        log_mel_spec = extract_log_mel_spectrogram_torch(waveform, self.to_mel_spec) #convert to logmelspec
        log_mel_spec = log_mel_spec.unsqueeze(0)
        #mean-variance normalization 
        if self.norm_status == 'byol':
            log_mel_spec = self.pre_norm(log_mel_spec)
               
        return log_mel_spec

    def __len__(self):
        return len(self.audio_files_list)

class DeepCluster_Reassigned(Dataset):

    def __init__(self,args,audio_file_list,label_list,audio_indexes,tfms):
        self.audio_files = audio_file_list
        self.audio_labels = label_list
        self.audio_indexes = audio_indexes
        self.dataset = self.make_dataset()
        self.aug = tfms
        self.length = args.length_wave
        self.to_mel_spec = MelSpectrogramLibrosa()

        
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
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        wave = torch.tensor(wave)
        
        waveform = extract_window_torch(self.length, wave) #extract a window
        log_mel_spec = extract_log_mel_spectrogram_torch(waveform, self.to_mel_spec) #convert to logmelspec
        log_mel_spec = log_mel_spec.unsqueeze(0)        
        log_mel_spec = self.aug(log_mel_spec.clone().detach()) #augmentation 
        return log_mel_spec,label

    def __len__(self):
        return len(self.audio_files)