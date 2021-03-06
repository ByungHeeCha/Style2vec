import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# from quadratum import transforms as qtrfm
import torchvision.transforms as T
from model import Style2VecV2, NegLossV2, Normalize
from data import PolyvoreDatasetv2
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import numpy as np
from tqdm import tqdm
import os
import timm
import timm.optim
import time

def resize_and_pad(img_size):
    def resize_and_pad_with_certain_size(img):
        h, w = img.size
        if w < h:
            resized_w = int(w*img_size/h)
            img = T.Resize((resized_w, img_size))(img)
            img = T.Pad((0, (img_size-resized_w)//2, 0, (img_size-resized_w) -
                         (img_size-resized_w)//2), padding_mode='constant', fill=255)(img)
        else:
            resized_h = int(h*img_size/w)
            img = T.Resize((img_size, resized_h))(img)
            img = T.Pad(((img_size-resized_h)//2, 0, (img_size -
                                                      resized_h)-(img_size-resized_h)//2, 0), padding_mode='constant', fill=255)(img)
        return img
    return resize_and_pad_with_certain_size

USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("No CPU!")
    exit()

def invert(img):
    return 1-img

transform = T.Compose([
                T.Lambda(resize_and_pad(240)),
                T.RandomRotation((-10, 10)),
                T.RandomResizedCrop(240, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(invert),
                # T.ToPILImage(),
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

train_dataset = PolyvoreDatasetv2("./data/train_filtered.json", "./data/images", transform=transform)
style_set_len = len(train_dataset)
print(style_set_len)

num_neg = 3

train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)

num_train_layer = -1

def train(model, loader):
    
    loss_fn = NegLossV2()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
    optimizer = timm.optim.AdamP(model.parameters())
    
    train_start = time.time()
    model.to(device=device)
    loss_fn.to(device=device)
    for epoch in range(1, 70+1):
        
        train_loss = 0
        epoch_start = time.time()
        pbar = tqdm(loader)
        for idx, (input_img, neg_img, labels) in enumerate(pbar):
            model.train()
            # model.cnn.train(False)
            if num_train_layer != -1:
                model.cnn._conv_stem.train(False)
                model.cnn._bn0.train(False)
                for block_index, block in enumerate(model.cnn._blocks):
                    if block_index < len(model.cnn._blocks) - num_train_layer:
                        block.train(False)
            optimizer.zero_grad()
            i = input_img.to(device=device, dtype=dtype).squeeze()
            n = neg_img.to(device=device, dtype=dtype).squeeze()
            labels = labels.flatten().cuda()
            ivec, cvec, logits = model.forward_img(i)
            loss = 0
            tot_loss = 0
            celoss = 0
            if logits is not None:
                loss = nn.CrossEntropyLoss(reduction='mean')(logits, labels) * 20
                loss.backward(retain_graph=True)
                celoss = loss.item() / 20
            for index in range(ivec.shape[0]):
                nvec = model.forward_neg(
                    n[(num_neg*(ivec.shape[0]-1)*index):(num_neg*(ivec.shape[0]-1)*(index+1))])
                loss = loss_fn(ivec[index], cvec[[a for a in range(ivec.shape[0]) if a!=index]], nvec)
                loss /= ivec.shape[0]
                loss.backward(retain_graph=(index != ivec.shape[0]-1))
                tot_loss += loss.item()
            train_loss += (tot_loss+celoss)
            # loss.backward()
            optimizer.step()
            # scheduler.step(epoch - 1 + idx / style_set_len)
            pbar.set_description("Loss %s, CE: %s" % (tot_loss, celoss))
            # if idx % 8 == 0:
            #     loss /= (8)
            #     train_loss += (loss.item()*8)
            #     loss.backward(retain_graph=False)
            #     optimizer.step()
            # elif idx == len(train_dataset)-1:
            #     loss /= ((idx-1) % 8 + 1)
            #     train_loss += (loss.item()*((idx-1) % 8 + 1))
            #     loss.backward(retain_graph=False)
            #     optimizer.step()
            # elif idx >= (len(train_dataset) // 8)*8:
            #     loss /= ((idx-1) % 8 + 1)
            #     train_loss += (loss.item()*((idx-1) % 8 + 1))
            #     loss.backward(retain_graph=True)
            # else:
            #     loss /= 8
            #     train_loss += (loss.item()*8)
            #     loss.backward(retain_graph=True)
            
        train_loss /= (idx + 1)
        
        epoch_time = time.time() - epoch_start
        print("Epoch\t", epoch, 
              "\tLoss\t", train_loss, 
              "\tTime\t", epoch_time,
             )
        model_save_name = 'Style2vecV2_B1_invert_head_linear_aug_adamp_num_train_layer_{}_emb_dim_{}_neg_{}_best.pt'.format(num_train_layer, 512, num_neg, epoch)
        path = F"./trained_model/{model_save_name}"
        if epoch == 1:
            torch.save(model.state_dict(), path)
            best_loss = train_loss
        elif best_loss > train_loss:
            torch.save(model.state_dict(), path)
    elapsed_train_time = time.time() - train_start
    print('Finished training. Train time was:', elapsed_train_time)


model = Style2VecV2(num_train_layer=num_train_layer, train_classification=False, num_class=train_dataset.get_num_class())
train(model, train_data)
