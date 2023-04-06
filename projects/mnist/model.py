import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, pmap, vmap
from jax.scipy.special import logsumexp


def init_mlp(layer_widths: list, parent_key, scale: float = 0.01):
    params = []
    keys = jax.random.split(parent_key, num=len(layer_widths) - 1)
    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        w_key, b_key = jax.random.split(key)
        params.append(
            [
                scale * jax.random.normal(w_key, shape=(out_width, in_width)),
                scale * jax.random.normal(b_key, shape=(out_width,)),
            ]
        )
    return params


def mlp_predict(params, x):
    hidden_layers = params[:-1]

    activation = x
    for w, b in hidden_layers:
        activation = jax.nn.relu(jnp.dot(w, activation) + b)

    w_last, b_last = params[-1]
    logits = jnp.dot(w_last, activation) + b_last
    return logits - logsumexp(logits)


batched_mlp_predict = vmap(mlp_predict, in_axes=(None, 0))
