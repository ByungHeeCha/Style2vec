import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import Style2VecV2, NegLossV2
from data import PolyvoreFITBDataset, PolyvoreFITBEvalDataset
from efficientnet_pytorch import EfficientNet
import collections
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
    T.Lambda(resize_and_pad(380)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def collate_seq(batch):
    """Return batches as we want: with variable item lengths."""
    return batch
    # if isinstance(batch[0], collections.Mapping):
    #     # return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    #     return batch

train_dataset = PolyvoreFITBDataset(
    "./data/train_no_dup.json", "./data/images", transform=transform)

test_dataset = PolyvoreFITBEvalDataset("./data/images", transform=transform)
train_data = DataLoader(train_dataset, batch_size=32,
                        shuffle=True, collate_fn=collate_seq)
test_data = DataLoader(test_dataset, batch_size=1, collate_fn = collate_seq)

num_train_layer = 3

def test(model, emb_model, loader):
    model.to(device=device)
    emb_model.to(device=device)
    acc = 0
    for idx, data in enumerate(test_data):
        model.eval()
        emb_model.eval()
        images = [d['question'] for d in data]
        cands = [d['candidates'] for d in data]
        img = images[0].to(device=device, dtype=dtype)
        cand = cands[0].to(device=device, dtype=dtype)
        blnk_pos = data[0]['blank_position']
        ans = data[0]['answer']
        with torch.no_grad():
            xvecs = emb_model.embedding(img).unsqueeze(1)
            candvecs = emb_model.embedding(cand)
            # xvecs = nn.utils.rnn.pack_sequence(xvecs)
            output, _ = model(xvecs)
            if blnk_pos == 0:
                f_logit = 0
            else:
                f_logit = nn.Softmax()(torch.mv(candvecs, output[blnk_pos-1, 0, :512]))
            
            # if blnk_pos == img.shape[0]:
            #     b_logit = 0
            # else:
            #     b_logit = nn.Softmax()(torch.mv(candvecs, output[blnk_pos, 0, 512:]))
            # print(f_logit+b_logit)
            _, pred_idx = torch.max((f_logit), 0)
            if pred_idx == ans:
                acc += 1
    print("Acc: ", acc/(idx+1), acc)
            # hidden, len_hidden = nn.utils.rnn.pad_packed_sequence(output)


def train(emb_model, loader):

    model = nn.LSTM(512, 512, bidirectional=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_start = time.time()
    model.to(device=device)
    emb_model.to(device=device)
    loss_fn.to(device=device)

    for epoch in range(1, 5+1):

        train_loss = 0
        epoch_start = time.time()
        pbar = tqdm(train_data)
        for idx, images in enumerate(pbar):
            model.train()
            emb_model.train(False)
            optimizer.zero_grad()
            ind = sorted(list(range(len(images))), key=lambda k: int(images[k].shape[0]), reverse=True)
            img = [images[i].to(device=device, dtype=dtype) for i in ind]
            
            with torch.no_grad():
                xvecs = [emb_model.embedding(im) for im in img]
            xvecs = nn.utils.rnn.pack_sequence(xvecs)

            output, _ = model(xvecs)

            hidden, len_hidden = nn.utils.rnn.pad_packed_sequence(output)
            # xvecs, _ = nn.utils.rnn.pad_packed_sequence(xvecs)
            x_values = xvecs.data
            # print(len_hidden)
            # print(xvecs.batch_sizes)
            # seq_lens = xvecs.batch_sizes
            # print(xvecs.shape)
            # print(len_hidden > 5)
            # print(int((len_hidden > 5).sum()))
            # print(xvecs.shape)
            loss = 0
            start = 0
            for i, seq_len in enumerate(len_hidden):
                hidden_front = hidden[0:seq_len-1, i, :512]
                hidden_back = hidden[1:seq_len, i, 512:]
                front_logits = torch.mm(hidden_front, x_values.permute(1, 0))
                back_logits = torch.mm(hidden_back, x_values.permute(1, 0))
                loss += (loss_fn(front_logits, torch.LongTensor(list(range(start+1, start+seq_len))).to(device=device)) +
                         loss_fn(back_logits, torch.LongTensor(list(range(start, start+seq_len-1))).to(device=device)))
                start += seq_len
            # for i in range(len_hidden[0]):
            #     hidden_front = hidden[i, 0:num_batch[i], :512]
            #     hidden_back = hidden[i, 0:num_batch[i], 512:]
            #     xs = xvecs[i, 0:num_batch[i], :]
            #     logits_front = torch.mm(hidden_front, xs.T)
            #     logits_back = torch.mm(hidden_back, xs.T)
            #     loss += (loss_fn(logits_front, torch.LongTensor(list(range(num_batch[i]))))+
            #              loss_fn(logits_back, torch.LongTensor(list(range(num_batch[i])))))
            loss /= len(len_hidden)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_description("Loss %s" % (loss.item()))

        train_loss /= (idx + 1)

        epoch_time = time.time() - epoch_start

        print("Epoch\t", epoch,
              "\tLoss\t", train_loss,
              "\tTime\t", epoch_time,
              )
        model_save_name = 'bilstm_{}.pt'.format(epoch)
        path = F"./trained_model/{model_save_name}"
        torch.save(model.state_dict(), path)
    elapsed_train_time = time.time() - train_start
    print('Finished training. Train time was:', elapsed_train_time)
    test(model, emb_model, test_data)


state = torch.load(
    "trained_model/Style2vecV2_pad_edge_train_context_num_train_layer_3_emb_dim_512_neg_5_epoch_5.pt")

model = Style2VecV2(num_train_layer=num_train_layer)
model.load_state_dict(state)

s = torch.load(
    "trained_model/lstm_5.pt")
lstm = nn.LSTM(512, 512, bidirectional=True)
lstm.load_state_dict(s)
test(lstm, model, test_data)
# train(model, train_data)
