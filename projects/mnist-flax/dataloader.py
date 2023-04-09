import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def get_transform(img):
    """Transfom img to np.array and normalize [0, 1]"""
    return np.expand_dims(np.array(img, np.float32), axis=2) / 255.0


def custom_collate_fn(batch):
    """Custom collate fn. Change torch.Tensor to np.ndarray"""
    data = list(zip(*batch))
    imgs = np.array(data[0])
    labels = np.array(data[1])
    return imgs, labels


def get_datasets():
    """Load MNIST train and test datasets"""
    train_ds = MNIST(".", train=True, download=True, transform=get_transform)
    test_ds = MNIST(".", train=False, download=True, transform=get_transform)
    return train_ds, test_ds


def get_dataloader(ds, train, batch_size):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=custom_collate_fn,
        drop_last=True,
    )
