import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import copy
import timm
import numpy as np
import torch.nn.functional as F


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Style2Vec(nn.Module):
    def __init__(self, num_train_layer=2, emb_dim=512):
        super(Style2Vec, self).__init__()
        self.cnn = EfficientNet.from_pretrained(
            'efficientnet-b4', advprop=True, include_top=False)
        self.mlp = nn.Linear(1792, emb_dim)
        self.context_mlp = nn.Linear(1792, emb_dim)
        for ct, child in enumerate(self.cnn.children()):
            if ct != 7 and ct != 2:
                for param in child.parameters():
                    param.requires_grad = False
            elif ct == 7:
                for param in child.parameters():
                    param.requires_grad = True
            elif ct == 2:
                for i, block in enumerate(child.children()):
                    if i >= 32-num_train_layer:
                        # block.apply(init_weights)
                        for param in block.parameters():
                            param.requires_grad = True
                    else:
                        for param in block.parameters():
                            param.requires_grad = False

    def forward(self, image, context):
        ivec = self.mlp(self.cnn(image).flatten(start_dim=1))
        contextvec = self.context_mlp(self.cnn(context).flatten(start_dim=1))
        return ivec, contextvec


class NegLoss(nn.Module):
    def __init__(self):
        super(NegLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, ivec, contextvec, label):
        return self.criterion((ivec * contextvec).sum(-1), label)


# class Style2VecV2(nn.Module):
#     def __init__(self, num_train_layer=2, mlp=nn.Linear(1280, 512), train_context=True):
#         super(Style2VecV2, self).__init__()
#         self.cnn = EfficientNet.from_pretrained(
#             'efficientnet-b1', advprop=True, include_top=False)
#         # self.mlp = nn.Sequential(nn.Linear(1280, emb_dim), MemoryEfficientSwish(), nn.Linear(emb_dim, emb_dim), nn.Tanh())
#         self.mlp = mlp
#         if train_context:
#             # self.context_mlp = nn.Linear(1280, emb_dim)
#             # self.context_mlp = nn.Sequential(
#             #     nn.Linear(1280, emb_dim), MemoryEfficientSwish(), nn.Linear(emb_dim, emb_dim), nn.Tanh())
#             self.context_mlp = copy.deepcopy(mlp)
#         else:
#             self.context_mlp = self.mlp

#         for param in self.cnn.parameters():
#             param.requires_grad = False
#         for param in self.cnn._conv_head.parameters():
#             param.requires_grad = True
#         for param in self.cnn._bn1.parameters():
#             param.requires_grad = True
#         for param in self.cnn._avg_pooling.parameters():
#             param.requires_grad = True
#         for i in range(1, num_train_layer+1):
#             for param in self.cnn._blocks[-i].parameters():
#                 param.requires_grad = True
#         # for ct, child in enumerate(self.cnn.children()):
#         #     if ct != 7 and ct != 2:
#         #         for param in child.parameters():
#         #             param.requires_grad = False
#         #     elif ct == 7:
#         #         for param in child.parameters():
#         #             param.requires_grad = True
#         #     elif ct == 2:
#         #         for i, block in enumerate(child.children()):
#         #             if i>=32-num_train_layer:
#         #                 for param in block.parameters():
#         #                     param.requires_grad = True
#         #             else:
#         #                 for param in block.parameters():
#         #                     param.requires_grad = False

#     def forward_img(self, image):
#         img_emb = self.cnn(image)
#         context_emb = self.context_mlp(img_emb.flatten(start_dim=1))
#         img_emb = self.mlp(img_emb.flatten(start_dim=1))
#         return img_emb, context_emb

#     def forward_neg(self, negs):
#         neg_emb = self.context_mlp(self.cnn(negs).flatten(start_dim=1))
#         return neg_emb

#     def forward(self, image, context):
#         ivec = self.mlp(self.cnn(image).flatten(start_dim=1))
#         contextvec = self.context_mlp(self.cnn(context).flatten(start_dim=1))
#         return ivec, contextvec

#     def embedding(self, image):
#         return self.mlp(self.cnn(image).flatten(start_dim=1))

class NegLossV2(nn.Module):
    def __init__(self, use_sim=False):
        super(NegLossV2, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.use_sim = use_sim

    def forward(self, ivec, contextvecs, negvecs):
        if self.use_sim:
            p = F.cosine_similarity(contextvecs, ivec.unsqueeze(0), dim=1)
            n = F.cosine_similarity(negvecs, ivec.unsqueeze(0), dim=1)
        else:
            p = torch.mv(contextvecs, ivec)
            n = torch.mv(negvecs, ivec)
        return self.criterion(torch.cat([p, n]), torch.cat([torch.ones_like(p), torch.zeros_like(n)]))


class Style2VecV2(nn.Module):
    def __init__(self, num_train_layer=2, mlp=nn.Linear(1280, 512), train_context=True, train_classification=False, num_class=121):
        super(Style2VecV2, self).__init__()
        self.cnn = EfficientNet.from_pretrained(
            'efficientnet-b1', advprop=True, include_top=False)
        # self.cnn = timm.create_model('efficientnet_b1',pretrained=True, features_only=True)
        self.mlp = mlp
        self.train_classification = train_classification
        if train_classification:
            self.classifier = nn.Linear(1280, num_class)
        if train_context:
            self.context_mlp = copy.deepcopy(mlp)
        else:
            self.context_mlp = self.mlp
        if num_train_layer != -1:
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.cnn._conv_head.parameters():
                param.requires_grad = True
            for param in self.cnn._bn1.parameters():
                param.requires_grad = True
            for param in self.cnn._avg_pooling.parameters():
                param.requires_grad = True
            for i in range(1, num_train_layer+1):
                for param in self.cnn._blocks[-i].parameters():
                    param.requires_grad = True
        else:
            for param in self.cnn.parameters():
                param.requires_grad = True

    def forward_img(self, image):
        img_feat = self.cnn(image)
        context_emb = self.context_mlp(img_feat.flatten(start_dim=1))
        img_emb = self.mlp(img_feat.flatten(start_dim=1))
        if self.train_classification:
            logits = self.classifier(img_feat.flatten(start_dim=1))
        else:
            logits = None
        return img_emb, context_emb, logits

    def forward_neg(self, negs):
        neg_emb = self.context_mlp(self.cnn(negs).flatten(start_dim=1))
        return neg_emb

    def forward(self, image, context):
        ivec = self.mlp(self.cnn(image).flatten(start_dim=1))
        contextvec = self.context_mlp(self.cnn(context).flatten(start_dim=1))
        return ivec, contextvec

    def embedding(self, image):
        return self.mlp(self.cnn(image).flatten(start_dim=1))


class Style2VecV3(nn.Module):
    def __init__(self, num_train_layer=2, emb_dim=512):
        super(Style2VecV3, self).__init__()
        self.cnn = EfficientNet.from_pretrained(
            'efficientnet-b1', advprop=True, num_classes=512)
        self.context_cnn = EfficientNet.from_pretrained(
            'efficientnet-b1', advprop=True, num_classes=512)
        if num_train_layer != -1:
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.cnn._conv_head.parameters():
                param.requires_grad = True
            for param in self.cnn._bn1.parameters():
                param.requires_grad = True
            for param in self.cnn._avg_pooling.parameters():
                param.requires_grad = True
            for i in range(1, num_train_layer+1):
                for param in self.cnn._blocks[-i].parameters():
                    param.requires_grad = True
        else:
            for param in self.cnn.parameters():
                param.requires_grad = True
            for param in self.context_cnn.parameters():
                param.requires_grad = True

    def forward_img(self, image):
        img_emb = self.cnn(image)
        context_emb = self.context_cnn(image)
        return img_emb, context_emb

    def forward_neg(self, negs):
        neg_emb = self.context_cnn(negs)
        return neg_emb

    def forward(self, image, context):
        ivec = self.cnn(image)
        contextvec = self.context_cnn(context)
        return ivec, contextvec

    def embedding(self, image):
        return self.cnn(image)
