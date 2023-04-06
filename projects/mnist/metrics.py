import jax.numpy as jnp
from model import batched_mlp_predict


def accuracy(params, dataset_imgs, dataset_gts):
    pred_classes = jnp.argmax(batched_mlp_predict(params, dataset_imgs), axis=1)
    return jnp.mean(dataset_gts == pred_classes)
