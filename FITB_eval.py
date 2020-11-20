import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import Style2Vec, Style2VecV2, NegLossV2
from data import PolyvoreFITBDataset, PolyvoreFITBEvalDataset
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import collections
import numpy as np
from tqdm import tqdm
import os
import glob


import time


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
            img = T.Pad(((img_size-resized_h)//2, 0, (img_size -
                                                      resized_h)-(img_size-resized_h)//2, 0), padding_mode='edge')(img)
        return img
    return resize_and_pad_with_certain_size


USE_GPU = True
dtype = torch.float32  # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("No CPU!")
    exit()

transform = T.Compose([
    T.Lambda(resize_and_pad(240)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def collate_seq(batch):
    """Return batches as we want: with variable item lengths."""
    return batch
    # if isinstance(batch[0], collections.Mapping):
    #     # return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    #     return batch



test_dataset = PolyvoreFITBEvalDataset("./data/images", transform=transform)
test_data = DataLoader(test_dataset, batch_size=64, collate_fn=collate_seq)

acc_dict = {}
def test(emb_model, loader):
    model.to(device=device)
    emb_model.to(device=device)
    acc = 0
    num = 0
    pbar = tqdm(test_data)
    for idx, data in enumerate(pbar):
        emb_model.eval()
        images = [d['question'] for d in data]
        cands = [d['candidates'] for d in data]
        img = [image.to(device=device, dtype=dtype) for image in images]
        cand = [c.to(device=device, dtype=dtype) for c in cands]
        ans = torch.LongTensor([d['answer']
                                for d in data]).to(device=device, dtype=dtype)
        num += len(data)
        with torch.no_grad():
            q_mean = torch.stack([emb_model.embedding(im).mean(0) for im in img])
            cand_vecs = torch.stack([emb_model.embedding(im) for im in cand])
            diff = torch.norm(q_mean.unsqueeze(1).repeat(1, 4, 1)-cand_vecs, dim=2)
            pred = torch.argmin(diff, dim=1)
            acc += (ans == pred).sum()
    print("acc: ", acc.item()/num)
    return acc.item()


    # hidden, len_hidden = nn.utils.rnn.pad_packed_sequence(output)
for filepath in glob.iglob("trained_model/Style2vecV2_B1_head_linear_emb_dim_512_neg_5_epoch_1.pt"):
    print(filepath)
    state = torch.load(
        filepath)
    # model = Style2VecV2(num_train_layer=0, mlp=nn.Sequential(nn.Linear(
    #     1280, 512), MemoryEfficientSwish(), nn.Linear(512, 512), nn.Tanh()))
    model = Style2VecV2(num_train_layer=0, train_context=True)
    model.load_state_dict(state)
    acc_dict[filepath] = test(model, test_data)


