from datasets.birdsong_dataset import BirdSongDataset
from datasets.tf_speech import TfSpeech
from datasets.libri100 import Libri100


def get_dataset(downstream_task_name):
    if downstream_task_name == "birdsong":
        return BirdSongDataset()
    elif downstream_task_name == "tf_speech":
        return TfSpeech()
    elif downstream_task_name == "libri_100":
        return Libri100()    
    else:
        raise NotImplementedError