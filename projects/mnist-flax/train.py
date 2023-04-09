import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from dataloader import get_dataloader, get_datasets
from flax.training import train_state
from model import CNN, apply_model, update_model


def train_epoch(state, dataloader):
    epoch_loss, epoch_accuracy = [], []
    for _, (imgs, labels) in enumerate(dataloader):
        grads, loss, accuracy = apply_model(state, imgs, labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def eval_epoch(state, dataloader):
    epoch_loss, epoch_accuracy = [], []
    for _, (imgs, labels) in enumerate(dataloader):
        _, loss, accuracy = apply_model(state, imgs, labels)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    eval_loss = np.mean(epoch_loss)
    eval_accuracy = np.mean(epoch_accuracy)
    return eval_loss, eval_accuracy


def create_train_state(rng, config):
    model = CNN()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(config.lr, config.momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_and_eval(config, seed: int = 0):
    train_ds, test_ds = get_datasets()
    train_dl = get_dataloader(train_ds, train=True, batch_size=config.batch_size)
    test_dl = get_dataloader(test_ds, train=False, batch_size=config.batch_size)
    rng = jax.random.PRNGKey(seed)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs):
        state, train_loss, train_acc = train_epoch(state, train_dl)
        test_loss, test_acc = eval_epoch(state, test_dl)
        
        logging.info(
            f"Train: epoch: {epoch}, train_loss: {train_loss}, train_accuracy: {train_acc*100}"
        )
        logging.info(f"Test: epoch: {epoch}, test_loss: {test_loss}, test_accuracy: {test_acc*100}")

    return state
