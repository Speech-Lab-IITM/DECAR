import torch
from torch import nn
import re
import sys
import logging
from pathlib import Path
import torch.nn.functional as F

'''
class DeepCluster_ICASSP(nn.Module):
    def __init__(self):
        super(DeepCluster_ICASSP, self).__init__()
        self.model_efficient = EfficientNet.from_name('efficientnet-b0',include_top = False, in_channels = 1,image_size = None)
        self.classifier = nn.Sequential(nn.Dropout(0.5),nn.Linear(1280, 512),nn.ReLU(),nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU())
        #self.classifier = nn.Sequential(nn.Dropout(0.5),nn.Linear(1280, 1280),nn.ReLU())
        self.top_layer = nn.Linear(512, 256)
        #self.top_layer = nn.Linear(1280, 512)
    def forward(self,batch):
        x = self.model_efficient(batch)
        x = x.flatten(start_dim=1) #1280 (already swished)
        x = self.classifier(x)
        
        if self.top_layer:
            x = self.top_layer(x)
        
        return x
'''
class NetworkCommonMixIn():
    """Common mixin for network definition."""

    def load_weight(self, weight_file, device):
        """Utility to load a weight file to a device."""

        state_dict = torch.load(weight_file, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove unneeded prefixes from the keys of parameters.
        weights = {}
        for k in state_dict:
            m = re.search(r'(^fc\.|\.fc\.|^features\.|\.features\.)', k)
            if m is None: continue
            new_k = k[m.start():]
            new_k = new_k[1:] if new_k[0] == '.' else new_k
            weights[new_k] = state_dict[k]
        # Load weights and set model to eval().
        self.load_state_dict(weights)
        self.eval()
        logging.info(f'Using audio embbeding network pretrained weight: {Path(weight_file).name}')
        return self

    def set_trainable(self, trainable=False):
        for p in self.parameters():
            p.requires_grad = trainable



class AudioNTT2020Task6(nn.Module, NetworkCommonMixIn):
    """DCASE2020 Task6 NTT Solution Audio Embedding Network."""

    def __init__(self, n_mels, d):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x):
        x = self.features(x)       # (batch, ch, mel, time)       
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x = self.fc(x)
        return x


class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    def __init__(self, args, out_dim, n_mels=64, d=512, nmb_prototypes=3000):
        super().__init__(n_mels=n_mels, d=d)
        self.args = args
        '''
        self.projection_head = nn.Sequential(
                nn.Linear(d, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, out_dim),
            )
        '''    
        self.classifier = nn.Sequential(nn.Dropout(0.5),nn.Linear(d, 512),nn.ReLU(),nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU())
        self.top_layer = nn.Linear(512, 256)
        #self.prototypes = nn.Linear(out_dim, nmb_prototypes[0], bias=False)
        

    def forward(self, batch):
        #print(batch.shape)
        z = super().forward(batch)

        (z1, _) = torch.max(z, dim=1)
        z2 = torch.mean(z, dim=1)
        z = z1 + z2

        x = self.classifier(z)

        if self.top_layer:
            x = self.top_layer(x)
        
        return x        