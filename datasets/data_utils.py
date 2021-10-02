import torch
import torchaudio
import librosa
import tensorflow as tf
import numpy as np
class DataUtils():

    root_dir ={
        "Birdsong" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/birdsong",
        "IEMOCAP" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/iemocap/iemocap/IEMOCAP",
        "Libri100" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/libri100/",
        "MusicalInstruments" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/magenta",
        "tut_urban" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/utu/TUT-urban-acoustic-scenes-2018-development",
        "voxceleb_v1" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/voxceleb/",
        "language_identification" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/lid"
    }

    @classmethod
    def map_labels(cls,label_array):
        uarray = np.unique(label_array)
        label_dict = dict()
        for i,label in enumerate(uarray):
            label_dict[label] = i
        return label_dict

    @classmethod
    def extract_log_mel_spectrogram(cls,audio_path,
                                sample_rate=16000,
                                frame_length=400,
                                frame_step=160,
                                fft_length=1024,
                                n_mels=64,
                                fmin=60.0,
                                fmax=7800.0):
        """Extract frames of log mel spectrogram from a raw waveform."""
        waveform,sr = librosa.core.load(audio_path, sample_rate)
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
        
        return np.expand_dims(log_mel_spectrograms.numpy(),axis=0)
    
    extract_mffc = torchaudio.transforms.MFCC(
                                            sample_rate=16000,
                                            n_mfcc=30,
                                            log_mels=True) 

    @classmethod
    def collate_fn_padd_2(cls,batch):
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

    @classmethod
    def read_mfcc(cls,filename):
        # print(filename)
        waveform, sample_rate = torchaudio.load(filename)   
        return torch.transpose(cls.extract_mffc(waveform),-2,-1) # change shape -> C,T,nfeats

    @classmethod
    def collate_fn_padd(cls,batch):
        '''
        Padds batch of variable length
        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## padd
        
        batch_x = [torch.squeeze(torch.Tensor(t)) for t,y in batch]
        batch_y = [y for t,y in batch]
        batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first = True)
        batch_x = batch_x.unsqueeze(1)
        batch_y = torch.Tensor(batch_y).type(torch.LongTensor)
        # print("batch y",batch_y)
        return batch_x,batch_y