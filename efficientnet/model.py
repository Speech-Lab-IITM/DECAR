
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn

class DeepCluster_ICASSP(nn.Module):
    def __init__(self, no_of_classes =256):
        super(DeepCluster_ICASSP, self).__init__()
        self.model_efficient = EfficientNet.from_name('efficientnet-b0',
                                                    include_top = False,
                                                    in_channels = 1,
                                                    image_size = None)
        self.classifier = nn.Sequential(
                            nn.Dropout(0.5),nn.Linear(1280, 512),nn.ReLU(),
                            nn.Dropout(0.5),nn.Linear(512, 512),nn.ReLU())
        self.top_layer = nn.Linear(512,no_of_classes)

    def forward(self,batch):
        x = self.model_efficient(batch)
        x = x.flatten(start_dim=1) #1280 (already swished)
        x = self.classifier(x)
        
        if self.top_layer:
            x = self.top_layer(x)
        
        return x