"""Independent small ImageNet CNN comparison; input remains RGB float [0,1]."""
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class MobileNetOOD(nn.Module):
 def __init__(self):
  super().__init__();self.features=mobilenet_v3_small(weights=None).features
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1));self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
  self.projection=nn.Linear(576,128);self.classifier=nn.Linear(128,2)
 def backbone_embedding(self,x):return self.features((x-self.mean)/self.std).mean((2,3))
 def forward_outputs(self,x):
  embedding=self.projection(self.backbone_embedding(x));return self.classifier(embedding),embedding
 def forward(self,x):return self.forward_outputs(x)[0]
