from tqdm import tqdm
import pandas as pd 

# /speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/evaluation_setup
# /speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/audio
# fold1_evaluate.txt  fold1_test.txt  fold1_train.txt


train_csv = pd.read_csv("/speech/Databases/Birdsong/TutUrban/TUT-urban-acoustic-scenes-2018-development/evaluation_setup/fold1_train.txt",
                        sep="\t", header=None)
train_csv.columns =  ['Path', 'Label']




