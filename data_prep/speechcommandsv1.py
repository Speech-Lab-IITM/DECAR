from tqdm import tqdm
import pandas as pd
import os

test_labels = ["yes", "no", "up", "down","left", "right", "on", "off", "stop", "go" ] #10
core_labels = test_labels + ["zero","one","two","three","four","five","six","seven","eight","nine"] # 20
auxliary_labels = ["bird","dog","happy", "wow","bed","cat","house","marvin","sheila","tree"] #10
labels = core_labels+ auxliary_labels

df = pd.DataFrame(columns = [ 'Path', 'Label'])
root_dir = "/speech/sandesh/icassp/tf_speech/train/audio"

for labe_id,label in tqdm(enumerate(labels)):
    audiofiles = os.listdir(os.path.join(root_dir,label))
    for file in tqdm(audiofiles ,leave=False):
        file_path = os.path.join(root_dir,label,file)
        df=df.append( {'Path' : file_path, 'Label' : label} , ignore_index = True)



df.to_csv('/speech/sandesh/icassp/tf_speech/train/labels.csv',index=False)