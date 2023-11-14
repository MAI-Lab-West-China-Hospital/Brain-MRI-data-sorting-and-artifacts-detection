from monai.networks.nets import EfficientNetBN
import torch
import torch.nn as nn
from torchsummary import summary

# print(torch.__version__)
class MyEfficientNet(nn.Module):
    def __init__(self, model_name, spatial_dims, in_channels, num_classes, pretrained, dropout_rate=0.2):
        super().__init__()

        self.backbone = EfficientNetBN(model_name,
                                       pretrained=pretrained,
                                       spatial_dims=spatial_dims,
                                       in_channels=in_channels,
                                       num_classes=num_classes,
                                       )

        self.backbone._fc = nn.Sequential(nn.Linear(in_features=1280, out_features=512),
                                          nn.Dropout(dropout_rate),
                                          nn.Linear(in_features=512, out_features=num_classes)
                                          )
       

    def forward(self, inputs: torch.Tensor):
        out = self.backbone(inputs)
        return out

