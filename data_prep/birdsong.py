from tqdm import tqdm
import pandas as pd
import os

root_dir = "/speech/Databases/Birdsong/BirdSong"

ff_txt = os.path.join(root_dir,"freefield1010/6035814")
df1 = pd.read_csv(ff_txt) 
df1['Path'] = df1.apply(lambda row: os.path.join( 'freefield1010','spec',str(row.itemid)+'.wav.npy'), axis = 1)
df1 = df1.drop(['itemid'], axis=1)
df1 = df1.reindex(columns=['Path','hasbird'])
df1.rename(columns = {'hasbird':'Lables'}, inplace = True)
df1.to_csv(os.path.join(root_dir,"freefield1010_data.csv"),index=False)


wab_txt = os.path.join(root_dir,"Warblr/6035817")
df2 = pd.read_csv(wab_txt) 
df2['Path'] = df2.apply(lambda row: os.path.join( 'Warblr','spec',str(row.itemid)+'.wav.npy'), axis = 1)
df2 = df2.drop(['itemid'], axis=1)
df2 = df2.reindex(columns=['Path','hasbird'])
df2.rename(columns = {'hasbird':'Lables'}, inplace = True)
df2.to_csv("/speech/Databases/Birdsong/BirdSong/Warblr_data.csv",index=False)
combined_df = df1.append(df2)
combined_df.to_csv(os.path.join(root_dir,"combined_data.csv"),index=False)