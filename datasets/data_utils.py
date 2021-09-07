import torch
import torchaudio

class DataUtils():
    
    extract_mffc = torchaudio.transforms.MFCC(
                                            sample_rate=16000,
                                            n_mfcc=30,
                                            log_mels=True) 

    @classmethod
    def read_mfcc(cls,filename):
        print(filename)
        waveform, sample_rate = torchaudio.load(filename)   
        return torch.transpose(cls.extract_mffc(waveform),-2,-1) # change shape -> C,T,nfeats

    @classmethod
    def collate_fn_padd(cls,batch):
        '''
        Padds batch of variable length
        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## padd
        
        batch_x = [torch.squeeze(torch.Tensor(t)) for t,y in batch]
        batch_y = [y for t,y in batch]
        batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first = True)
        batch_x = batch_x.unsqueeze(1)
        batch_y = None#torch.Tensor(batch_y).type(torch.LongTensor)

        return batch_x,batch_y