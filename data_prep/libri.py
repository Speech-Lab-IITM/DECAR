# /speech/srayan/icassp/downstream/data/libri100_spkid
# 585 speakers
# /5708 test
# /22830 train
#  test_split.txt  train_split.txt  utt2spk

from tqdm import tqdm
import pandas as pd

train_txt = "/speech/srayan/icassp/downstream/data/libri100_spkid/train_split.txt"
test_txt = "/speech/srayan/icassp/downstream/data/libri100_spkid/test_split.txt"
train_ids = []
test_ids = []


# with open(test_txt) as f:
#     test_ids = f.read().splitlines() 

utt2spk_root = "/speech/srayan/icassp/downstream/data/libri100_spkid/utt2spk"
utt2spk = dict()
  

with open(utt2spk_root) as f:
    for x in tqdm(f):
        key , value = x.split()
        if utt2spk.get(key, None)==None:
            utt2spk[key] = value #.add(1, 'Geeks')
        else:
            raise NotImplementedError    

print("no_of_speakers",len(set(utt2spk.values())))


with open(test_txt) as f:
    test_ids = f.read().splitlines() 


test_csv = pd.DataFrame(columns = ['Path', 'Label'])
test_speakers=[]
for i in tqdm(test_ids):
    test_speakers.append(utt2spk[i])
    test_csv = test_csv.append({'Path' : i, 'Label' : utt2spk[i] }, 
                ignore_index = True)
print("no_of_speakers",len(set(test_speakers)))

test_csv.to_csv("test_data.csv", index=False)

with open(train_txt) as f:
    train_ids = f.read().splitlines() 


train_csv = pd.DataFrame(columns = ['Path', 'Label'])
train_speakers=[]
for i in tqdm(train_ids):
    train_speakers.append(utt2spk[i])
    train_csv = train_csv.append({'Path' : i, 'Label' : utt2spk[i] }, 
                ignore_index = True)
print("no_of_speakers",len(set(train_speakers)))
train_csv.to_csv("train_data.csv", index=False)