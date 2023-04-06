import logging

import jax
import jax.numpy as jnp
from config import Config
from dataloader import get_data, get_dataloader, prepare_dataset
from jax import grad, jit, value_and_grad
from metrics import accuracy
from model import batched_mlp_predict, init_mlp


def loss_fn(params, imgs, gt_labels):
    predictions = batched_mlp_predict(params, imgs)
    return -jnp.mean(predictions * gt_labels)


@jit
def update(params, imgs, gt_labels, lr=0.01):
    loss, grads = value_and_grad(loss_fn)(params, imgs, gt_labels)
    return loss, jax.tree_map(lambda p, g: p - lr * g, params, grads)


def train(model, config, train_ds, test_ds):
    # Prepare data
    train_dl = get_dataloader(train_ds, config.BATCH_SIZE)
    train_imgs, train_labels = get_data(train_ds)
    test_imgs, test_labels = get_data(test_ds)

    # Training loop
    for epoch in range(config.EPOCHS):
        for idx, (imgs, labels) in enumerate(train_dl):
            gt_labels = jax.nn.one_hot(labels, len(train_ds.classes))
            loss, model = update(model, imgs, gt_labels)
            if idx % 100 == 0:
                logging.info(f"Loss: {loss}")

        # Metrics
        train_acc = accuracy(model, train_imgs, train_labels)
        test_acc = accuracy(model, test_imgs, test_labels)

        # Epoch log
        print(f"Epoch: {epoch} train acc: {train_acc} test acc: {test_acc}")
    return model


if __name__ == "__main__":
    config = Config()
    rng = jax.random.PRNGKey(config.SEED)

    # Create dataset
    train_ds, test_ds = prepare_dataset()

    # Init MLP
    model = init_mlp(config.ARCHITECTURE, rng)

    # Train
    model = train(model, config, train_ds, test_ds)
