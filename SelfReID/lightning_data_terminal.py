# Torch and Torchvision stuff
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Sampler, BatchSampler

# Lightning stuff
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule

# Misc stuff
import os
import importlib
import numpy as np
import random
from sklearn.model_selection import KFold


class TimestampSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last:bool = False, random_time_per_epoch:bool=False, shuffle:bool=True):
        """
        Ensures that each batch contains masks from only one timestamp.
        Args:
            data_source (list): a list of tuples [(embd, path)]
            batch_size (int): Number of samples per batch.
        Kwargs:
            drop_last (bool)
            random_time_per_epoch (bool): Flag to configure if selecting randomised timestamps every batch
            shuffle (bool): shuffle each ITEM under certain timestamp
        """
        self.seed = int(torch.empty((), dtype=torch.int64).random_().item())
        random.seed(self.seed)
        self.dataset = data_source
        print(type(self.dataset))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_entry = random_time_per_epoch
        self.timestamps = list(self.dataset._get_timestamps_info().values())
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
                # target = [random.choice(class_list) for _ in range(self.batch_size)]
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

class TimestampBatchSampler(BatchSampler):
    def __init__(self, data_source, shuffle:bool=True):
        """
        Ensures that each batch contains masks from only one timestamp.
        Args:
            data_source (list): a list of tuples [(embd, path)]
            batch_size (int): Number of samples per batch.
        Kwargs:
            drop_last (bool)
            num_timestamp (int): Sample from how many subfolders
        """
        self.seed = int(torch.empty((), dtype=torch.int64).random_().item())
        random.seed(self.seed)

        self.data_source = data_source
        self.timestamps = list(self.data_source._get_timestamps_info().values())
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.timestamps)))
        if self.shuffle:
            random.shuffle(self.timestamps)
        for idx in indices:
            yield self.timestamps[idx]


    def __len__(self):
        return len(self.timestamps)

class HybridBatchSampler(BatchSampler):
    def __init__(self, data_source, shuffle:bool=True):
        """
        Assume the dataset is now <date>/<timestamp>/<image_id>_<label>.jpeg;
        Ensures that each batch contains masks from only one timestamp within one date.
        Args:
            data_source (list): a list of tuples [(embd, path)]
            batch_size (int): Number of samples per batch.
        Kwargs:
            drop_last (bool)
        """
        self.seed = int(torch.empty((), dtype=torch.int64).random_().item())
        random.seed(self.seed)

        self.data_source = data_source
        self.entries = list(self.data_source._get_hybrid_info().values())
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.entries)))
        if self.shuffle:
            random.shuffle(self.entries)
        for idx in indices:
            yield self.entries[idx]

    def __len__(self):
        return len(self.entries)


def custom_collate_fn(batch):
    masks, paths = zip(*batch)

    # return torch.utils.data.default_collate(images), torch.utils.data.default_collate(class_ids), torch.utils.data.default_collate(dates)
    return torch.utils.data.default_collate(masks), torch.utils.data.default_collate(paths)

# def custom_collate_fn(batch):
#     # embd1, embd2, paths = zip(*batch)
#     embd1 = [item[0] for item in batch]
#     embd2 = [item[1] for item in batch]
#     paths = [item[2] for item in batch]
#
#     embd1 = torch.cat(embd1, dim=0)
#     embd2 = torch.cat(embd2, dim=0)
#
#     return embd1, embd2, paths


class SelfReIDDataModule(LightningDataModule):
    def __init__(self, dataset_name, size, root, predict_root, batch_size, num_workers, k_fold=False, k=1):
        super(SelfReIDDataModule, self).__init__()
        self.dataset_prefix = getattr(importlib.import_module(f'data.{dataset_name}.{dataset_name}'), dataset_name)
        exec(f"self.size = {size}")
        all_folders = sorted([os.path.join(root, item) for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))])
        self.masks_dict = {"train":[], "val":[], "test": [], "predict": []}
        if k_fold:
            kf = KFold(n_splits=10, shuffle=True, random_state=84)
            all_folder_indices = list(range(len(all_folders)))
            train_idx, test_idx = list(kf.split(all_folder_indices))[k-1]
            print("-" * 20 + f"KFold = {k}" + "-" * 20)
            train_folders = [all_folders[i] for i in train_idx]
            test_folders = [all_folders[i] for i in test_idx]

            self.masks_dict["train"] = self.collect_images(train_folders)
            self.masks_dict["test"] = self.collect_images(test_folders)
        else:
            split_length = int((3/5) * len(all_folders))
            train_folders = all_folders[:split_length]
            test_folders = all_folders[split_length:int((4/5)*len(all_folders))]
            train_folders.extend(all_folders[int((4/5)*len(all_folders)):])
            self.masks_dict["train"] += self.collect_images(train_folders)
            self.masks_dict["test"] += self.collect_images(test_folders)

        predict_folders = sorted([os.path.join(predict_root, item) for item in os.listdir(predict_root) if os.path.isdir(os.path.join(predict_root, item))])
        self.masks_dict["predict"] = self.collect_images(predict_folders)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def retrieve_dataset(self, type):
        return self.dataset_prefix(masks=self.masks_dict[type], size=self.size)

    def collect_images(self, folders):
        result = []
        for folder in folders:
            result.extend(sorted([os.path.join(folder, item) for item in os.listdir(folder)]))
        return result

    def setup(self, stage):
        self.train_set = self.retrieve_dataset(type="train")
        self.val_set = self.retrieve_dataset(type="train")
        self.test_set = self.retrieve_dataset(type="test")
        self.predict_set = self.retrieve_dataset(type="predict")

        # self.test_set_train_data = self.retrieve_dataset(type='train', pseudo=False)
        # self.test_set_test_data = self.retrieve_dataset(type='test', pseudo=False)

        # self.predict_set_train_data = self.retrieve_dataset(type='train', pseudo=False)
        # self.predict_set_test_data = self.retrieve_dataset(type='test', pseudo=False)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_sampler=TimestampBatchSampler(data_source=self.train_set, shuffle=True),
                          #batch_sampler=TimestampBatchSampler(data_source=self.train_set, batch_size=self.batch_size),
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          sampler=None,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          sampler=None,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_set,
                          sampler=None,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_fn)

class MultiSelfReIDDataModule(LightningDataModule):
    def __init__(self, dataset_name, size, root, batch_size, num_workers, k=1):
        super(MultiSelfReIDDataModule, self).__init__()
        self.dataset_prefix = getattr(importlib.import_module(f'data.{dataset_name}.{dataset_name}'), dataset_name)
        exec(f"self.size = {size}")

        # all_folders = sorted([os.path.join(root, item) for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))])
        self.masks_dict = {"train":[], "val":[], "test": [], "predict": []}
        train_split, val_split, test_split = self.KFold_split(root, k=k)
        self.masks_dict["train"] += self.collect_images(train_split)
        self.masks_dict["val"] += self.collect_images(val_split)
        self.masks_dict["test"] += self.collect_images(test_split)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def KFold_split(self, root, k=1):
        date_folders = sorted(os.listdir(root))
        assert (1 <= k <= len(date_folders))
        train_split, val_split, test_split = [], [], []
        for id, item in enumerate(date_folders):
            if id + k == len(date_folders):
                test_split.append(os.path.join(root, item))
            elif ((id + k) % len(date_folders) + 1) == len(date_folders):
                val_split.append(os.path.join(root, item))
            else:
                train_split.append(os.path.join(root, item))

        return train_split, val_split, test_split

    def retrieve_dataset(self, type):
        return self.dataset_prefix(masks=self.masks_dict[type], size=self.size)

    def collect_images(self, folders):
        result = []
        for folder in folders:
            all_subfolders = sorted(os.listdir(folder))
            half_subfolders = all_subfolders[:(len(all_subfolders)//3)]
            for subfolder in half_subfolders:
                base_folder = os.path.join(folder, subfolder)
                result.extend(sorted([os.path.join(base_folder, item) for item in os.listdir(base_folder)]))
        return result

    def setup(self, stage):
        self.train_set = self.retrieve_dataset(type="train")
        self.val_set = self.retrieve_dataset(type="val")
        self.test_set = self.retrieve_dataset(type="test")
        # self.predict_set = self.retrieve_dataset(type="predict")

        # self.test_set_train_data = self.retrieve_dataset(type='train', pseudo=False)
        # self.test_set_test_data = self.retrieve_dataset(type='test', pseudo=False)

        # self.predict_set_train_data = self.retrieve_dataset(type='train', pseudo=False)
        # self.predict_set_test_data = self.retrieve_dataset(type='test', pseudo=False)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_sampler=HybridBatchSampler(data_source=self.train_set, shuffle=True),
                          #batch_sampler=TimestampBatchSampler(data_source=self.train_set, batch_size=self.batch_size),
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          sampler=None,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          sampler=None,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_fn)

    # def predict_dataloader(self):
    #     return DataLoader(self.predict_set,
    #                       sampler=None,
    #                       batch_size=self.batch_size,
    #                       shuffle=False,
    #                       num_workers=self.num_workers,
    #                       collate_fn=custom_collate_fn)
