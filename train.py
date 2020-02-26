import math
import os

from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint

from dataset import get_train_data
from model import get_model
from utils import AnnealingCallback, ex

# Ignores TensorFlow CPU messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@ex.capture
def schedule(epoch, initial_learning_rate, lr_decay_start_epoch):
    """Defines exponentially decaying learning rate."""

    if epoch < lr_decay_start_epoch:
        return initial_learning_rate
    else:
        return initial_learning_rate * math.exp((10 * initial_learning_rate) * (lr_decay_start_epoch - epoch))


@ex.automain
def train(weights_path, epochs, batch_size, initial_epoch,
          kl_start_epoch, kl_alpha_increase_per_epoch):
    """Trains a model."""
    print ('loading data...')
    # Loads or creates training data.
    input_shape, train, valid, train_targets, valid_targets = get_train_data()
    print ('getting model...')
    # Loads or creates model.
    model, checkpoint_path, kl_alpha = get_model(input_shape,
                                        scale_factor=len(train)/batch_size,
                                        weights_path=weights_path)

    # Sets callbacks.
    checkpointer = ModelCheckpoint(checkpoint_path, verbose=1,
                                   save_weights_only=True, save_best_only=True)

    scheduler = LearningRateScheduler(schedule)
    annealer = Callback() if kl_alpha is None else AnnealingCallback(kl_alpha, kl_start_epoch, kl_alpha_increase_per_epoch)

    print ('fitting model...')
    # Trains model.
    model.fit(train, train_targets, batch_size, epochs,
              initial_epoch=initial_epoch,
              callbacks=[checkpointer, scheduler, annealer],
              validation_data=(valid, valid_targets))

