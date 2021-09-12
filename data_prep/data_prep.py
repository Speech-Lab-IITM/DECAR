import os 
import numpy as np
import tensorflow as tf 
import librosa
from tqdm import tqdm
from multiprocessing import Pool


root_dir = "/speech/sandesh/icassp/deep_cluster/birdsong/wav"

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_log_mel_spectrogram(audio_path,
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
        
        return log_mel_spectrograms.numpy()


def write_feats(files_array):
    for file in tqdm(files_array):
        feat = extract_log_mel_spectrogram(os.path.join(dir,file))
        np.save(os.path.join('/speech/sandesh/icassp/deep_cluster/birdsong/wav','spec',file),feat)
        break
  
def run_parallel():
    create_dir(os.path.join(root_dir,'spec'))
    
    list_files = np.array(os.listdir(dir))
    list_ranges = np.array_split(list_files, 10)
  
    pool = Pool(processes=len(list_ranges))
    pool.map(write_feats, list_ranges)
  
# Driver code
if __name__ == '__main__':
    run_parallel()