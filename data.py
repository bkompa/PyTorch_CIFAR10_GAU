import os
import zipfile
import torch
import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from itertools import filterfalse

import matplotlib.pyplot as plt
import numpy as np 


def plot_example(dataset, idx):
    plt.imshow(np.transpose(dataset[idx].numpy(),(1,2,0)).squeeze())
    plt.show()

class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x): 
        return TF.rotate(x, self.angle)

class RollTransform:
    def __init__(self, pixels, axis):
        self.pixels = pixels
        self.axis = axis

    def __call__(self, x): 
        return torch.roll(x, self.pixels, dims=self.axis)

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.train_dataset = None

    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform, download=True)

        if self.train_dataset is None:
            self.train_dataset = dataset
        val_split = self.hparams.val_split
        split_indices = list(filterfalse(lambda x: 10000*val_split <= x < 10000*(val_split+1), range(50000)))

        train_data_split = Subset(dataset, split_indices)
        dataloader = DataLoader(
            train_data_split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        # make sure self.train_dataset is not None
        self.train_dataloader()

        val_split = self.hparams.val_split
        split_indices = range(10000*val_split, 10000*(val_split+1))

        val_data_split = Subset(self.train_dataset, split_indices)
        dataloader = DataLoader(
            val_data_split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def get_validation_data(self):
        val_dataloader = self.val_dataloader()
        imgs = torch.stack([img for img, _ in val_dataloader.dataset])
        labels = torch.Tensor([label for _, label in val_dataloader.dataset])
        return imgs, labels

    def get_rotation_data(self, rotation_angle=0):
        transform = T.Compose([RotationTransform(rotation_angle), T.ToTensor(), T.Normalize(self.mean, self.std)])
        cifar10_data = CIFAR10(root=self.hparams.data_dir, train=False, download=True, transform=None)
        transform_data = torch.stack([transform(img) for img, label in cifar10_data])
        labels = torch.Tensor([label for img, label in cifar10_data])

        return transform_data, labels

    def get_roll_data(self, roll_pixels=0, axis=2):
        transform = T.Compose([T.ToTensor(), RollTransform(roll_pixels, axis), T.Normalize(self.mean, self.std)])
        cifar10_data = CIFAR10(root=self.hparams.data_dir, train=False, download=True, transform=None)
        transform_data = torch.stack([transform(img) for img, label in cifar10_data])
        labels = torch.Tensor([label for img, label in cifar10_data])

        return transform_data, labels
