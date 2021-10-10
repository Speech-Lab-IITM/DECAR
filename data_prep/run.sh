python data_prep.py --root_dir "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/birdsong/freefield1010" --no_workers 10 --prefix "wav" --suffix "spec"
python data_prep.py --root_dir "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/birdsong/Warblr" --no_workers 10 --prefix "wav" --suffix "spec"


python data_prep.py --root_dir "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/SpeechCommandsV1/train" --no_workers 10 --prefix "audio" --suffix "spec" --file "labels.csv"