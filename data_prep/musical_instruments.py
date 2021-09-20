import json
import pandas as pd
import os
from tqdm import tqdm
# root_dir :: nsynth-train(289205) , nsynth-valid 


root_dir = "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/magenta"

fi = open(os.path.join(root_dir,'nsynth-train','examples.json'))
train_json = json.load(fi)

train_csv=pd.DataFrame(columns=['Path','Label'])
for key in tqdm(train_json.keys()):
    train_csv=train_csv.append({'Path' : os.path.join('nsynth-train','spec',key+'.wav.npy'),
     'Label' : train_json[key]['instrument_family'] },ignore_index=True)
train_csv.to_csv(os.path.join(root_dir,'train_data.csv'),index=False)
fi.close()



fi = open(os.path.join(root_dir,'nsynth-valid','examples.json'))
valid_json = json.load(fi)

valid_csv=pd.DataFrame(columns=['Path','Label'])
for key in tqdm(valid_json.keys()):
    valid_csv=valid_csv.append({'Path' : os.path.join('nsynth-valid','spec',key+'.wav.npy'),
     'Label' : valid_json[key]['instrument_family'] },ignore_index=True)
valid_csv.to_csv(os.path.join(root_dir,'valid_data.csv'),index=False)
fi.close()

fi = open(os.path.join(root_dir,'nsynth-test','examples.json'))
valid_json = json.load(fi)

valid_csv=pd.DataFrame(columns=['Path','Label'])
for key in tqdm(valid_json.keys()):
    valid_csv=valid_csv.append({'Path' : os.path.join('nsynth-test','spec',key+'.wav.npy'),
     'Label' : valid_json[key]['instrument_family'] },ignore_index=True)
valid_csv.to_csv(os.path.join(root_dir,'test_data.csv'),index=False)
fi.close()



