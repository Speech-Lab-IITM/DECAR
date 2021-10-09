import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset

from specaugment import specaug

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


class DeepCluster(Dataset):

    def __init__(self, data_dir_list):
        self.audio_files_list = data_dir_list

    def __getitem__(self, idx):
        audio_file = self.audio_files_list[idx]
        #wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        #log_mel_spec = extract_log_mel_spectrogram(wave)
        log_mel_spec = np.load(audio_file).tolist()       
        return log_mel_spec

    def __len__(self):
        return len(self.audio_files_list)

class DeepCluster_Reassigned(Dataset):

    def __init__(self,audio_file_list,label_list,audio_indexes):
        self.audio_files = audio_file_list
        self.audio_labels = label_list
        self.audio_indexes = audio_indexes
        self.dataset = self.make_dataset()

        
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
        log_mel_spec = torch.tensor(np.load(audio_file).tolist())
        log_mel_spec = specaug(log_mel_spec.clone().detach()) 
        return log_mel_spec,label

    def __len__(self):
        return len(self.audio_files)