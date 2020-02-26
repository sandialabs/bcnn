import os

import numpy as np
from sacred import Experiment
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import binary_crossentropy
import tensorflow_probability as tfp

ex = Experiment()
ex.add_config("configs/toy_config.json")


def round_down(num, factor):
    """Rounds num to next lowest multiple of factor."""

    return (num // factor) * factor


def acc(a, b):
    """Calculates number of matches in two np arrays."""
    return np.count_nonzero(a == b) / a.size


def absolute_file_paths(directory, match=""):
    """Gets absolute file paths from a directory.

    Does not include subdirectories.

    Args:
        match: Returns only paths of files containing the given string.
    """
    paths = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if match in f:
                paths.append(os.path.abspath(os.path.join(root, f)))
        break
    return paths


def get_latest_file(directory, match=""):
    """Gets the absolute file path of the last modified file in a directory.

    Args:
        match: Returns only paths of files containing the given string.
    """

    paths = absolute_file_paths(directory, match=match)
    if paths:
        return max(paths, key=os.path.getctime)
    else:
        return None


def standardize(raw):
    """Transforms data to have mean 0 and std 1."""

    return (raw - np.mean(raw)) / np.std(raw)


def variational_free_energy_loss(model, scale_factor, kl_alpha):
    """Defines variational free energy loss.

    Sum of KL divergence (supplied by tfp) and binary cross-entropy.
    """

    # KL Divergence should be applied once per epoch only, so
    # scale_factor should be num_samples / batch_size.
    kl = sum(model.losses) / scale_factor

    def loss(y_true, y_pred):
        bce = binary_crossentropy(y_true, y_pred)
        return bce + K.get_value(kl_alpha) * kl

    return loss


@ex.capture
def normal_prior(prior_std):
    """Defines normal distribution prior for Bayesian neural network."""

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        tfd = tfp.distributions
        dist = tfd.Normal(loc=tf.zeros(shape, dtype),
                          scale=dtype.as_numpy_dtype((prior_std)))
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior_fn

class AnnealingCallback(Callback):
    def __init__(self, kl_alpha, kl_start_epoch, kl_alpha_increase_per_epoch):
        self.kl_alpha = kl_alpha
        self.kl_start_epoch = kl_start_epoch
        self.kl_alpha_increase_per_epoch = kl_alpha_increase_per_epoch
    def on_epoch_end (self, epoch, logs={}):
        if epoch >= self.kl_start_epoch - 2:
            new_kl_alpha = min(K.get_value(self.kl_alpha) + self.kl_alpha_increase_per_epoch, 1.)
            K.set_value(self.kl_alpha, new_kl_alpha)
        print ("Current KL Weight is " + str(K.get_value(self.kl_alpha)))
