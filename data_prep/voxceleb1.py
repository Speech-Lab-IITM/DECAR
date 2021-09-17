import os 
import numpy as np
import tensorflow as tf 
import librosa
from tqdm import tqdm
from multiprocessing import Pool
import argparse
from functools import partial
import pandas as pd
tf.config.set_visible_devices([], 'GPU')

train_csv = pd.read_csv("/speech/Databases/Birdsong/Voxceleb1/dev/test_vox.csv")
train_csv['Path'] = train_csv.apply(lambda row: os.path.join('spec',str(str(row.file_path)[46:]+'.npy')) , axis = 1)
train_csv  = train_csv.drop(['file_path'],axis=1)
train_csv = train_csv.reindex(columns=['Path','label'])
train_csv.to_csv("/speech/Databases/Birdsong/Voxceleb1/dev/test_data.csv",index=False)

# def get_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--root_dir', type=str,
#                         help='down_stream task name')
#     parser.add_argument('--no_workers', default=10, type=int, metavar='N',
#                         help='number of total epochs to run')
#     parser.add_argument('--prefix', default='wav', type=str,
#                         help='number of total epochs to run')                    
#     return parser


# def create_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)




# def extract_log_mel_spectrogram(audio_path,
#                                 sample_rate=16000,
#                                 frame_length=400,
#                                 frame_step=160,
#                                 fft_length=1024,
#                                 n_mels=64,
#                                 fmin=60.0,
#                                 fmax=7800.0):
#         """Extract frames of log mel spectrogram from a raw waveform."""
#         waveform,sr = librosa.core.load(audio_path, sample_rate)
#         stfts = tf.signal.stft(
#             waveform,
#             frame_length=frame_length,
#             frame_step=frame_step,
#             fft_length=fft_length)
#         spectrograms = tf.abs(stfts)

#         num_spectrogram_bins = stfts.shape[-1]
#         lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
#         linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
#             num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
#             upper_edge_hertz)
#         mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
#         mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
#             linear_to_mel_weight_matrix.shape[-1:]))

#         mel_spectrograms = tf.clip_by_value(
#             mel_spectrograms,
#             clip_value_min=1e-5,
#             clip_value_max=1e8)

#         log_mel_spectrograms = tf.math.log(mel_spectrograms)

#         return log_mel_spectrograms.numpy()


# def write_feats(root_dir,prefix,files_array):
#     for file in tqdm(files_array):
#         # print("-------------------")
#         # print(root_dir,prefix,file)
#         if file.endswith("wav"):
#             feat = extract_log_mel_spectrogram(os.path.join(root_dir,prefix,file))
#             create_dir(os.path.dirname(os.path.join(root_dir,'spec',file)))
#             np.save(os.path.join(root_dir,'spec',file),feat)

# def run_parallel(args):
#     create_dir(os.path.join(args.root_dir,'spec'))

#     # trim = len("/speech/Databases/Birdsong/Voxceleb1/dev/wav/")
#     df_csv = pd.read_csv("/speech/Databases/Birdsong/Voxceleb1/test_vox.csv")
#     df_csv['audio_path'] = df_csv.apply(lambda row: os.path.join(str(row.file_path)[46:]), axis = 1)

#     # test_csv = pd.read_csv("/speech/Databases/Birdsong/Voxceleb1/test_vox.csv")
#     # trim = len("/speech/Databases/Birdsong/Voxceleb1/test/wav/")
#     # test_csv['audio_path'] = valid_csv.apply(lambda row: os.path.join(str(row.file_path)[trim:]), axis = 1)
#     # # args.root_dir = ""


#     list_files = np.array(df_csv['audio_path'].values)
#     list_ranges = np.array_split(list_files, args.no_workers)
#     pfunc=partial(write_feats,args.root_dir,args.prefix)
#     pool = Pool(processes=len(list_ranges))
#     pool.map(pfunc, list_ranges)

# # Driver code
# if __name__ == '__main__':
#     parser = get_parser()
#     args = parser.parse_args()
#     print(args)
#     run_parallel(args) 