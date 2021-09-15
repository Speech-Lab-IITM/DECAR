import json
import pandas as pd
import os
from tqdm import tqdm
# need to cut file ???

root_dir = "/speech/Databases/Birdsong/IEMOCAP/"

full_datacsv = pd.read_csv(os.path.join(root_dir,'iemocap_final.csv'))
print(full_datacsv)