import math
import os

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from dataset import get_train_data
from model import get_model
from utils import ex

# Ignores TensorFlow CPU messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@ex.capture
def schedule(epoch, initial_learning_rate):
    """Defines exponentially decaying learning rate."""

    if epoch < 10:
        return initial_learning_rate
    else:
        return initial_learning_rate * math.exp((10 * initial_learning_rate) * (10 - epoch))


@ex.automain
def train(weights_path, epochs, batch_size, initial_epoch):
    """Trains a model."""
    print ('loading data...')
    # Loads or creates training data.
    input_shape, train, valid, train_targets, valid_targets = get_train_data()
    print ('getting model...')
    # Loads or creates model.
    model, checkpoint_path = get_model(input_shape,
                                       scale_factor=len(train)/batch_size,
                                       weights_path=weights_path)

    # Sets callbacks.
    checkpointer = ModelCheckpoint(checkpoint_path, verbose=1,
                                   save_weights_only=True, save_best_only=True)

    scheduler = LearningRateScheduler(schedule)
    print ('fitting model...')
    # Trains model.
    model.fit(train, train_targets, batch_size, epochs,
              initial_epoch=initial_epoch,
              callbacks=[checkpointer, scheduler],
              validation_data=(valid, valid_targets))

