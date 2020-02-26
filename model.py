import os

import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

from bayesian_unet import bayesian_unet
from bayesian_vnet import bayesian_vnet
from dropout_unet import dropout_unet
from dropout_vnet import dropout_vnet
from utils import ex, get_latest_file, variational_free_energy_loss


@ex.capture
def load_model(input_shape, weights_path, net, prior_std,
               kernel_size, activation, padding, num_gpus):
    """Loads model from .h5 file.

    If model is saved as multi-gpu, re-saves it as single-gpu.
    """

    # Loads model as multi-gpu, if possible.
    try:
        model = net(input_shape,
                    kernel_size=kernel_size,
                    activation=activation,
                    padding=padding,
                    prior_std=prior_std)
        model = multi_gpu_model(model, gpus=num_gpus)

        # Converts .h5 file to single-gpu.
        model.load_weights(weights_path)
        model = model.layers[-2]
        model.save_weights(weights_path)
    except ValueError as e:
        pass

    # Loads single-gpu model.
    model = net(input_shape,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                prior_std=prior_std)
    model.load_weights(weights_path)

    return model


@ex.capture
def get_model(input_shape, weights_dir, resume, bayesian,
              vnet, prior_std, kernel_size, activation, padding,
              kl_alpha, kl_start_epoch, kl_alpha_increase_per_epoch,
              ensemble, num_gpus, initial_epoch,
              scale_factor=1, weights_path=None):
    """Loads or creates model.

    If a weights path is specified, loads from that path. Otherwise, loads
    the most recently modified model.
    """

    os.makedirs(weights_dir + "/bayesian", exist_ok=True)
    os.makedirs(weights_dir + "/dropout", exist_ok=True)

    # Sets variables for ensemble model.
    if ensemble:
        checkpoint_path = (weights_dir + "/ensemble/ensemble-{epoch:02d}"
        "-{val_acc:.3f}-{val_loss:.0f}.h5")

        if weights_path:
            latest_weights_path = weights_path
        else:
            latest_weights_path = get_latest_file(weights_dir + "/bayesian")

        net = ensemble_vnet

    # Sets variables for bayesian model.
    elif bayesian:
        checkpoint_path = (weights_dir + "/bayesian/bayesian-{epoch:02d}"
        "-{val_acc:.3f}-{val_loss:.0f}.h5")

        if weights_path:
            latest_weights_path = weights_path
        else:
            latest_weights_path = get_latest_file(weights_dir + "/bayesian")

        net = bayesian_vnet if vnet else bayesian_unet

    # Sets variables for dropout model.
    else:
        checkpoint_path = (weights_dir + "/dropout/dropout-{epoch:02d}"
        "-{val_acc:.3f}-{val_loss:.2f}.h5")

        if weights_path:
            latest_weights_path = weights_path
        else:
            latest_weights_path = get_latest_file(weights_dir + "/dropout")

        net = dropout_vnet if vnet else dropout_unet

    # Loads or creates model.
    if latest_weights_path and resume:
        model = load_model(input_shape, latest_weights_path, net)
    else:
        model = net(input_shape,
                    kernel_size=kernel_size,
                    activation=activation,
                    padding=padding,
                    prior_std=prior_std)

    # Prints model summary.
    model.summary(line_length=127)

    # Converts to multi-gpu model if applicable.
    if num_gpus > 1:
        model = multi_gpu_model(model, gpus=num_gpus)

    # Sets loss function.
    if bayesian:
        if initial_epoch >= kl_start_epoch:
            kl_alpha = min(1., kl_alpha + (initial_epoch - kl_start_epoch) * kl_alpha_increase_per_epoch)

        kl_alpha = K.variable(kl_alpha)
        loss = variational_free_energy_loss(model, scale_factor, kl_alpha)
    else:
        kl_alpha = None
        loss = binary_crossentropy

    # Compiles model with Adam optimizer.
    model.compile(loss=loss,
                  optimizer=Adam(),
                  metrics=["accuracy"])

    return model, checkpoint_path, kl_alpha

