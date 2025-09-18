import os
import random
import numpy as np
from PIL import Image
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch.utils import data

# Mask dataset
class MaskDataset(data.Dataset):
    def __init__(self, masks, size=(256, 256)):
        self.masks = masks
        # self.masks_dict = {"train":[], "test":[]}
        # self.root = root
        # total_length = len(os.listdir(root))
        # if k_fold:
        #     train_items = os.listdir(root)[:]
        #     test_items = os.listdir(root)[:]
        # else:
        #     train_length = int((2/3)*total_length)
        #     train_items = os.listdir(root)[:train_length]
        #     test_items = os.listdir(root)[train_length:]

        # for item in os.listdir(root):
        #     sub_folder = os.path.join(root, item)
        #     for img in os.listdir(sub_folder):
        #         self.masks.append(os.path.join(sub_folder, img))
        # for item in train_items:
        #     sub_folder = os.path.join(root, item)
        #     for img in os.listdir(sub_folder):
        #         self.masks_dict["train"].append(os.path.join(sub_folder, img))
        #
        # for item in test_items:
        #     sub_folder = os.path.join(root, item)
        #     for img in os.listdir(sub_folder):
        #         self.masks_dict["test"].append(os.path.join(sub_folder, img))

        self.transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # self.transform2 = transforms.Compose([
        #     transforms.RandomResizedCrop(self.size, scale=(0.95, 1.0), ratio=(0.95,1.05)),
        #     transforms.Resize(self.size),
        #     transforms.ElasticTransform(alpha=100.0),
        #     # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        #     # transforms.RandomHorizontalFlip(),
        #     # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        #     # transforms.RandomGrayscale(p=0.2),
        #     transforms.GaussianBlur(3),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        mask_path = self.masks[index]
        mask = Image.open(mask_path).convert('RGB')
        mask = self.transform(mask)

        return mask, mask_path

    # def __getitem__(self, idx):
    #     if isinstance(idx, list):
    #         batch = [self.__getitem__(id) for id in idx]
    #         mask1_batch = torch.stack([b[0] for b in batch])
    #         mask2_batch = torch.stack([b[1] for b in batch])
    #         path_batch = [b[2] for b in batch]
    #         # print(mask1_batch.shape)
    #         return mask1_batch, mask2_batch, path_batch
    #     else:
    #         mask = Image.open(self.masks[idx])
    #         embd1 = self.transform1(mask)
    #         # neg = Image.open(self.find_neg_idx(self.masks[idx]))
    #         mask = torch.from_numpy(np.array(mask, dtype="float32")).permute(2, 0, 1)
    #         # neg = torch.from_numpy(np.array(neg, dtype="float32")).permute(2, 0, 1)
    #         embd2 = self.transform2(mask)
    #         return embd1, embd2, self.masks[idx]

    def find_neg_idx(self, mask_path):
        random.seed(datetime.now().timestamp())
        timestamp = mask_path.split("/")[-2]
        candidates = [item for item in self.masks if (timestamp in item and item != mask_path)]
        return random.choice(candidates)

    def _get_timestamps_info(self):
        # timestamps_info = {timestamp: [] for timestamp in os.listdir(self.root)}
        timestamps_info = {}
        for i, mask in enumerate(self.masks):
            timestamp_cur = mask.split("/")[-2]
            if timestamp_cur in timestamps_info.keys():
                timestamps_info[timestamp_cur].append(i)
            else:
                timestamps_info.update({timestamp_cur: []})

        return timestamps_info

    def _get_hybrid_info(self):
        timestamps_info = {}
        for i, mask in enumerate(self.masks):
            timestamp_cur = mask.split("/")[-2]
            date_cur = mask.split("/")[-3]
            entry = (date_cur, timestamp_cur)
            if entry in timestamps_info.keys():
                timestamps_info[entry].append(i)
            else:
                timestamps_info.update({entry: []})

        return timestamps_info
