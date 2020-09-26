import os
import json
import collections
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random

class PolyvoreDataset(Dataset):
    """ Polyvore dataset."""

    def __init__(self, json_file, img_dir, num_neg_samples=5, transform=None):
        """
        Args:
            json_file (string): Path to the json file with the data.
            img_dir (string): Directory where the image files are located.
            transform (callable, optional): Optional transform to be applied on
                                            a sample.
        """
        self.img_dir = img_dir
        self.data = json.load(open(json_file))
        self.processed_data = []
        print("Initializing "+json_file+" dataset")
        for idx, style in enumerate(self.data):
            for i in range(1, len(style["items"])+1):
                for j in range(i+1, len(style["items"])+1):
                    self.processed_data += [(style["set_id"]+"/"+str(i), style["set_id"]+"/"+str(j), 1),
                                            (style["set_id"]+"/"+str(j), style["set_id"]+"/"+str(i), 1)]
                for _ in range(num_neg_samples):
                    while True:
                        neg_outfit_index = random.randrange(len(self.data) - 1)
                        if neg_outfit_index != idx:
                            break
                    neg_i = random.randrange(1, len(self.data[neg_outfit_index]["items"])+1)
                    neg_set_id = self.data[neg_outfit_index]['set_id']
                    self.processed_data.append((style["set_id"]+"/"+str(i), neg_set_id+"/"+str(neg_i), -1))
        print("Done Initializing")
        self.transform = transform

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        """Get a specific index of the dataset (for dataloader batches).
        Args:
            idx: index of the dataset.
        Returns:
            Dictionary with two fields: images and texts, containing the corresponent sequence.
        """
        
        
        input_img = Image.open(os.path.join(self.img_dir, '%s.jpg' % self.processed_data[idx][0])).convert('RGB')
        neg_img = Image.open(os.path.join(self.img_dir, '%s.jpg' % self.processed_data[idx][1])).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            neg_img = self.transform(neg_img)
        
        return input_img, neg_img, self.processed_data[idx][2]

