import json
import pandas as pd
import os
from tqdm import tqdm
from pydub import AudioSegment  
import matplotlib.pyplot as plt


root_dirs = {
    "Birdsong" : "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/birdsong",
    "IEMOCAP"  : "/speech/Databases/Birdsong/IEMOCAP/",
    "MusicalInstruments" : "/speech/Databases/Birdsong/MusicalInstruments",
}


root_dir = root_dirs["Birdsong"]

ff_txt = os.path.join(root_dir,"freefield1010/6035814")
df1 = pd.read_csv(ff_txt) 
df1['Path'] = df1.apply(lambda row: os.path.join( 'freefield1010','specl2',str(row.itemid)+'.wav.npy'), axis = 1)
df1['AudioPath'] = df1.apply(lambda row: os.path.join( 'freefield1010','wav',str(row.itemid)+'.wav'), axis = 1)
df1.rename(columns = {'hasbird':'Label'}, inplace = True)
wab_txt = os.path.join(root_dir,"Warblr/6035817")
df2 = pd.read_csv(wab_txt) 
df2['Path'] = df2.apply(lambda row: os.path.join( 'Warblr','specl2',str(row.itemid)+'.wav.npy'), axis = 1)
df2['AudioPath'] = df2.apply(lambda row: os.path.join( 'Warblr','wav',str(row.itemid)+'.wav'), axis = 1)
df2.rename(columns = {'hasbird':'Label'}, inplace = True)
combined_df = df1.append(df2)

df1 = df1.reindex(columns=['Path','Label'])
df1.to_csv(os.path.join(root_dir,"freefield1010_data_l2.csv"),index=False)
df2 = df2.reindex(columns=['Path','Label'])
df2.to_csv(os.path.join(root_dir,"Warblr_data_l2.csv"),index=False)
combined_df = combined_df.reindex(columns=['Path','Label'])
combined_df.to_csv(os.path.join(root_dir,"combined_data_l2.csv"),index=False)