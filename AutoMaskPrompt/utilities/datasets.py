import os
import random
import numpy as np
from PIL import Image
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler

# Mask dataset
class MaskDataset(Dataset):
    def __init__(self, root="./images/RGB_masks/truth", size=64):
        self.masks = []
        self.root = root
        self.size = (size, size)
        for item in os.listdir(root):
            sub_folder = os.path.join(root, item)
            for img in os.listdir(sub_folder):
                self.masks.append(os.path.join(sub_folder, img))

        self.transform1 = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize(self.size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(self.size, scale=(0.95, 1.0), ratio=(0.95,1.05)),
            transforms.Resize(self.size),
            transforms.ElasticTransform(alpha=100.0),
            # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            # transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self.__getitem__(id) for id in idx]
            mask1_batch = torch.stack([b[0] for b in batch])
            mask2_batch = torch.stack([b[1] for b in batch])
            path_batch = [b[2] for b in batch]
            # print(mask1_batch.shape)
            return mask1_batch, mask2_batch, path_batch
        else:
            mask = Image.open(self.masks[idx])
            # neg = Image.open(self.find_neg_idx(self.masks[idx]))
            mask = torch.from_numpy(np.array(mask, dtype="float32")).permute(2, 0, 1)
            # neg = torch.from_numpy(np.array(neg, dtype="float32")).permute(2, 0, 1)
            embd1, embd2 = self.transform1(mask), self.transform2(mask)
            return embd1, embd2, self.masks[idx]

    def find_neg_idx(self, mask_path):
        random.seed(datetime.now().timestamp())
        timestamp = mask_path.split("/")[-2]
        candidates = [item for item in self.masks if (timestamp in item and item != mask_path)]
        return random.choice(candidates)

    def _get_timestamps_info(self):
        timestamps_info = {timestamp: [] for timestamp in os.listdir(self.root)}
        for i, mask in enumerate(self.masks):
            timestamp_cur = mask.split("/")[-2]
            timestamps_info[timestamp_cur].append(i)

        return timestamps_info

class TimestampBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last:bool = False, random_time_per_epoch:bool=False, shuffle:bool=True):
        """
        Ensures that each batch contains masks from only one timestamp.
        Args:
            data_source (list): a list of tuples [(embd1, embd2, path)]
            batch_size (int): Number of samples per batch.
        Kwargs:
            drop_last (bool)
            random_time_per_epoch (bool): Flag to configure if selecting randomised timestamps every batch
            shuffle (bool): shuffle each ITEM under certain timestamp
        """
        self.seed = int(torch.empty((), dtype=torch.int64).random_().item())
        random.seed(self.seed)

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_entry = random_time_per_epoch
        self.timestamps = list(self.data_source._get_timestamps_info().values())
        self.chosen_timestamp = random.sample(self.timestamps, 1)[0] if self.random_entry else None
        self.shuffle = shuffle

    def __iter__(self):
        if self.random_entry:
            chosen_class = self.chosen_timestamp
            if self.shuffle:
                random.shuffle(chosen_class)
            for i in range(0, len(self.chosen_timestamp), self.batch_size):
                target = chosen_class[i:i+self.batch_size]
                yield target
                # for idx in target:
                #     yield idx
                # if len(target) == self.batch_size or not self.drop_last:
                #     yield target
        else:
            for class_list in self.timestamps:
                if self.shuffle:
                    random.shuffle(class_list)
                for i in range(0, len(class_list), self.batch_size):
                    target = class_list[i:i+self.batch_size]
                    yield target
                    # for idx in target:
                    #     yield idx
                    # if len(target) == self.batch_size or not self.drop_last:
                    #     yield target

    def __len__(self):
        if self.random_entry:
            if self.drop_last:
                return int(len(self.chosen_timestamp) // self.batch_size)
            return int((len(self.chosen_timestamp) + self.batch_size - 1) // self.batch_size)
        else:
            if self.drop_last:
                return int(sum([len(entry) // self.batch_size for entry in self.timestamps()]))
            return int(sum([(len(entry) + self.batch_size - 1) // self.batch_size for entry in self.timestamps()]))

def custom_collate_fn(batch):
    # embd1, embd2, paths = zip(*batch)
    embd1 = [item[0] for item in batch]
    embd2 = [item[1] for item in batch]
    paths = [item[2] for item in batch]

    embd1 = torch.cat(embd1, dim=0)
    embd2 = torch.cat(embd2, dim=0)

    return embd1, embd2, paths
