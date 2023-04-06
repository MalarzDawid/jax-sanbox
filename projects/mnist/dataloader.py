import os

import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def get_data(ds):
    imgs = jnp.array(ds.data).reshape(len(ds), -1)
    labels = jnp.array(ds.targets)
    return imgs, labels


def custom_transform(img):
    return np.ravel(np.array(img, dtype=np.float32))


def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))
    labels = np.array(transposed_data[1])
    imgs = np.array(transposed_data[0])
    return imgs, labels


def prepare_dataset():
    train_dataset = MNIST(
        root="/tmp/mnist", train=True, download=True, transform=custom_transform
    )
    test_dataset = MNIST(
        root="/tmp/mnist", train=False, download=True, transform=custom_transform
    )
    return train_dataset, test_dataset


def get_dataloader(ds, batch_size: int = 32):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        drop_last=True,
    )
