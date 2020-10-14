import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import Style2VecV2, NegLossV2
from data import PolyvoreDatasetv2
from efficientnet_pytorch import EfficientNet
import numpy as np
from tqdm import tqdm
import os

import time

def resize_and_pad(img_size):
    def resize_and_pad_with_certain_size(img):
        w, h = img.size
        if w < h:
            resized_w = int(w*img_size/h)
            img = T.Resize((resized_w, img_size))(img)
            img = T.Pad((0, (img_size-resized_w)//2, 0, (img_size-resized_w)-(img_size-resized_w)//2))(img)
        else:
            resized_h = int(h*img_size/w)
            img = T.Resize((img_size, resized_h))(img)
            img = T.Pad(((img_size-resized_h)//2, 0, (img_size-resized_h)-(img_size-resized_h)//2, 0))(img)
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
                T.Lambda(resize_and_pad(380)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

train_dataset = PolyvoreDatasetv2("./data/train_no_dup.json", "./data/images", transform=transform)
style_set_len = len(train_dataset)
print(style_set_len)

train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)

num_train_layer = 3

def train(model, loader):
    
    loss_fn = NegLossV2()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_start = time.time()
    model.to(device=device)
    loss_fn.to(device=device)
    for epoch in range(1, 5+1):
        
        train_loss = 0
        epoch_start = time.time()
        pbar = tqdm(train_data)
        for idx, (input_img, neg_img) in enumerate(pbar):
            model.train()
            # model.cnn.train(False)
            model.cnn._conv_stem.train(False)
            model.cnn._bn0.train(False)
            for block_index, block in enumerate(model.cnn._blocks):
                if block_index < len(model.cnn._blocks) - num_train_layer:
                    block.train(False)
            optimizer.zero_grad()
            i = input_img.to(device=device, dtype=dtype).squeeze()
            n = neg_img.to(device=device, dtype=dtype).squeeze()
            ivec, cvec = model.forward_img(i)
            loss = 0
            for index in range(ivec.shape[0]):
                nvec = model.forward_neg(n[(5*index):(5*(index+1))])
                loss += loss_fn(ivec[index], cvec[[a for a in range(ivec.shape[0]) if a!=index]], nvec)
            loss /= ivec.shape[0]
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            pbar.set_description("Loss %s" % (loss.item()))
            
        train_loss /= (idx + 1)
        
        epoch_time = time.time() - epoch_start
        print("Epoch\t", epoch, 
              "\tLoss\t", train_loss, 
              "\tTime\t", epoch_time,
             )
        model_save_name = 'Style2vecV2_num_train_layer_{}_emb_dim_{}_neg_{}_epoch_{}.pt'.format(num_train_layer, 512, 5, epoch)
        path = F"./trained_model/{model_save_name}"
        torch.save(model.state_dict(), path)
    elapsed_train_time = time.time() - train_start
    print('Finished training. Train time was:', elapsed_train_time)

model = Style2VecV2(num_train_layer=num_train_layer)
train(model, train_data)