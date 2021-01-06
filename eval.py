import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import Style2Vec, NegLoss, Style2VecV2
from data import PolyvoreDataset
from efficientnet_pytorch import EfficientNet
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import time
import json
import random


def resize_and_pad(img_size):
    def resize_and_pad_with_certain_size(img):
        w, h = img.size
        if w < h:
            resized_w = int(w*img_size/h)
            img = T.Resize((resized_w, img_size))(img)
            img = T.Pad((0, (img_size-resized_w)//2, 0, (img_size-resized_w) -
                         (img_size-resized_w)//2), padding_mode='edge')(img)
        else:
            resized_h = int(h*img_size/w)
            img = T.Resize((img_size, resized_h))(img)
            img = T.Pad(((img_size-resized_h)//2, 0, (img_size-resized_h) -
                         (img_size-resized_h)//2, 0), padding_mode='edge')(img)
        return img
    return resize_and_pad_with_certain_size

USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("No CPU!")
    exit()

transform = T.Compose([
                # T.Lambda(resize_and_pad(380)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

data = json.load(open("./data/train_no_dup.json"))
# image = Image.open(os.path.join("./data/images", data[1]["set_id"],  '%s.jpg' % data[1]["items"][0]['index'])).convert('RGB')
# pos = Image.open(os.path.join("./data/images", data[1]["set_id"],  '%s.jpg' % data[1]["items"][1]['index'])).convert('RGB')

# neg = Image.open(os.path.join("./data/images", data[5]["set_id"],  '%s.jpg' % data[5]["items"][2]['index'])).convert('RGB')

image = Image.open(os.path.join(
    "./data/images", "119704139",  '4.jpg')).convert('RGB')
pos = Image.open(os.path.join(
    "./data/images", "119704139",  '2.jpg')).convert('RGB')

neg = Image.open(os.path.join(
    "./data/images", "148511719",  '1.jpg')).convert('RGB')

image = torch.unsqueeze(transform(image), 0)
pos = torch.unsqueeze(transform(pos), 0)
neg = torch.unsqueeze(transform(neg), 0)

image = image.cuda()
pos = pos.cuda()
neg = neg.cuda()

state = torch.load(
    "trained_model/Style2vecV2_pad_edge_train_context_num_train_layer_3_emb_dim_512_neg_5_epoch_2.pt")
model = Style2VecV2(num_train_layer=3)
model.load_state_dict(state)
model.cuda()
model.eval()
# print((model.embedding(image)*model.embedding(pos)).sum())
# print((model.embedding(image)*model.embedding(neg)).sum())
print(nn.CosineSimilarity()(model.embedding(image), model.embedding(pos)))
print(nn.CosineSimilarity()(model.embedding(image), model.embedding(neg)))

print((model.embedding(image)-model.embedding(pos)).norm())
print((model.embedding(image)-model.embedding(neg)).norm())

