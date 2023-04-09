import jax
import train
from absl import app, logging
from config import Config


def main(argv):
    config = Config()
    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    train.train_and_eval(config)


if __name__ == "__main__":
    app.run(main)
