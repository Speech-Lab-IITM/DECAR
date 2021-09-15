from pandas.core.indexes.base import Index
from tqdm import tqdm
import pandas as pd 
import os
# audio/ evaluation_setup/ 
# /speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/evaluation_setup
# /speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/audio
# fold1_evaluate.txt  fold1_test.txt  fold1_train.txt
# No Header
# audio/airport-barcelona-0-0-a.wav       airport

root_dir = "/speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development"
train_csv = pd.read_csv(os.path.join(root_dir,"evaluation_setup/fold1_train.txt"),
                        sep="\t", header=None)
train_csv.columns =  ['audio', 'Label']

train_csv['Path'] = train_csv.apply(lambda row: os.path.join('spec',str(row.audio).split('/')[-1]+'.npy'), 
                                                    axis = 1)
train_csv = train_csv.drop(['audio'], axis=1)
train_csv = train_csv.reindex(columns=['Path','Label'])
train_csv.to_csv(os.path.join(root_dir,"train_data.csv"),index=False)

valid_csv = pd.read_csv(os.path.join(root_dir,"evaluation_setup/fold1_evaluate.txt"),
                        sep="\t", header=None)
valid_csv.columns =  ['audio', 'Label']

valid_csv['Path'] = valid_csv.apply(lambda row: os.path.join('spec',str(row.audio).split('/')[-1]+'.npy'), 
                                                    axis = 1)
valid_csv = valid_csv.drop(['audio'], axis=1)
valid_csv = valid_csv.reindex(columns=['Path','Label'])
valid_csv.to_csv(os.path.join(root_dir,"valid_data.csv"),index=False)