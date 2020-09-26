import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish

import numpy as np
import torch.nn.functional as F


class Style2Vec(nn.Module):
    def __init__(self, num_train_layer=2, emb_dim=512):
        super(Style2Vec, self).__init__()
        self.cnn = EfficientNet.from_pretrained('efficientnet-b4', advprop=True, include_top=False)
        self.mlp = nn.Sequential(
            nn.Linear(1792, 1792),
            MemoryEfficientSwish(),
            nn.Linear(1792, emb_dim)
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(1792, 1792),
            MemoryEfficientSwish(),
            nn.Linear(1792, emb_dim)
        )
        # ct = 0
        for ct, child in enumerate(self.cnn.children()):
            # print(child)
            if ct != 7 and ct != 2:
                for param in child.parameters():
                    param.requires_grad = False
            elif ct == 7:
                for param in child.parameters():
                    param.requires_grad = True
            elif ct == 2:
                # i = 0
                for i, block in enumerate(child.children()):
                    if i>=32-num_train_layer:
                        # block.apply(init_weights)
                        for param in block.parameters():
                            param.requires_grad = True
                    else:
                        for param in block.parameters():
                            param.requires_grad = False
                    # i += 1
            # ct+=1
    
    def forward(self, image, context):
        # a = self.cnn(image)
        # print(a.shape)
        # print(a.flatten(start_dim=1).shape)
        ivec = self.mlp(self.cnn(image).flatten(start_dim=1))
        contextvec = self.context_mlp(self.cnn(context).flatten(start_dim=1))
        return ivec, contextvec

class NegLoss(nn.Module):
    def __init__(self):
        super(NegLoss, self).__init__()

    def forward(self, ivec, contextvec, label):
        ivec = ivec.view(-1, 1, ivec.shape[1])
        contextvec = contextvec.view(-1, contextvec.shape[1], 1)
        return -(label * torch.bmm(ivec, contextvec).squeeze()).sigmoid().log().sum(-1)