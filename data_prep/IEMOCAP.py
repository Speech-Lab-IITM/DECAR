import json
import pandas as pd
import os
from tqdm import tqdm
# need to cut file ???

root_dir = "/speech/Databases/Birdsong/IEMOCAP/"

full_datacsv = pd.read_csv(os.path.join(root_dir,'iemocap_final.csv'))
full_datacsv['Path'] = full_datacsv.apply(lambda row: os.path.join('spec',str(row.wav_file)+'.wav.npy'), axis = 1)
full_datacsv['Label'] = full_datacsv['emotion'] 
full_datacsv.drop(['index','start_time','end_time','val' ,'act' ,'dom' ,'emotion', 'wav_file'],axis=1,inplace=True)
full_datacsv = full_datacsv.reindex(columns=['Path','Label'])
full_datacsv.to_csv(os.path.join(root_dir,'data.csv'),index=False)