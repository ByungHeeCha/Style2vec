import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import Style2Vec, NegLoss
from data import PolyvoreDataset
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

train_dataset = PolyvoreDataset("./data/train_no_dup.json", "./data/images", transform=transform)

train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)

num_train_layer = 5

def train(model, loader):
    
    loss_fn = NegLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    # after = optim.lr_scheduler.CosineAnnealingLR(optimizer, 180)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20, after_scheduler=after)
    
    train_start = time.time()
    model.to(device=device)
    loss_fn.to(device=device)
    for epoch in range(1, 3+1):
        
        train_loss = 0
        epoch_start = time.time()
        pbar = tqdm(train_data)
        for idx, (input_img, target_img, label) in enumerate(pbar):
            model.train()
            model._conv_stem.train(False)
            model._bn0.train(False)
            for block_index, block in enumerate(model._blocks):
                if block_index < len(model._blocks) - num_train_layer:
                    block.train(False)
            optimizer.zero_grad()
            # print(input_img.shape)
            i = input_img.to(device=device, dtype=dtype)
            t = target_img.to(device=device, dtype=dtype)
            l = label.to(device=device, dtype=torch.long)


            ivec, tvec = model(i, t)
            # print(ivec.shape)
            loss = loss_fn(ivec, tvec, l)
            # print(loss)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        # train_loss /= (idx + 1)
        # scheduler.step()
        
        epoch_time = time.time() - epoch_start
        print("Epoch\t", epoch, 
              "\tLoss\t", train_loss, 
              "\tTime\t", epoch_time,
             )
        
    elapsed_train_time = time.time() - train_start
    print('Finished training. Train time was:', elapsed_train_time)

model = Style2Vec(num_train_layer=num_train_layer)
train(model, train_data)
model_save_name = 'Style2vec_num_train_layer_{}_emb_dim_{}_neg_{}_epoch_{}.pt'.format(num_train_layer, 512, 5, 3)
path = F"./trained_model/{model_save_name}"
torch.save(model.state_dict(), path)