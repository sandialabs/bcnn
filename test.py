from math import ceil
import os

import colorcet as cc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from dataset import add_chunk_to_arr, get_test_data, reconstruct
from model import get_model
from utils import ex, round_down

# Ignores TensorFlow CPU messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@ex.capture
def plots(bayesian, predict_dir, images_dir, lower_percentile, upper_percentile):
    prefix = "/bayesian/bayesian_" if bayesian else "/dropout/dropout_"

    sigmoids = np.load(predict_dir + prefix + "sigmoids.npy")
    pred = np.load(predict_dir + prefix + "pred.npy")
    test = np.load(predict_dir + "/test.npy")
    test_targets = np.load(predict_dir + "/test_targets.npy")

    lower, upper = np.percentile(sigmoids, [lower_percentile, upper_percentile], axis=0)
    unc = upper - lower

    # Plots four slices from each output numpy array.
    four_slices = range(test.shape[0] // 5, test.shape[0], test.shape[0] // 5)
    for i in four_slices:
        pred_slice = pred[i, :, :].squeeze()
        unc_slice = unc[i, :, :].squeeze()
        trg = test_targets[i, :, :].squeeze()
        img = test[i, :, :].squeeze()

        # Adds color bar to uncertainty map and saves.
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(unc_slice, cmap=cc.cm.CET_L19)
        fig.colorbar(im, cax=cax, orientation="vertical")
        plt.savefig(images_dir + prefix + "unc_{}.png".format(i))
        plt.close()

        # Saves image plots.
        plt.imsave(images_dir + prefix + "pred_{}.png".format(i),
                   pred_slice, cmap="Greys")
        plt.imsave(images_dir + "/img_{}.png".format(i), img, cmap="Greys")
        plt.imsave(images_dir + "/target_{}.png".format(i), trg, cmap="Greys")


@ex.capture
def save_predictions(sigmoids, pred, test, test_targets,
                     bayesian, predict_dir, images_dir):
    """Saves results of predictions."""

    os.makedirs(predict_dir + "/bayesian", exist_ok=True)
    os.makedirs(predict_dir + "/dropout", exist_ok=True)
    os.makedirs(images_dir + "/bayesian", exist_ok=True)
    os.makedirs(images_dir + "/dropout", exist_ok=True)

    prefix = "/bayesian/bayesian_" if bayesian else "/dropout/dropout_"

    # Saves output numpy arrays.
    np.save(predict_dir + prefix + "sigmoids.npy", sigmoids)
    np.save(predict_dir + prefix + "pred.npy", pred)
    np.save(predict_dir + "/test.npy", test)
    np.save(predict_dir + "/test_targets.npy", test_targets)

    # Plots four slices from each output numpy array.
    #four_slices = range(test.shape[0] // 5, test.shape[0], test.shape[0] // 5)
    #for i in four_slices:
        #pred_slice = pred[i, :, :].squeeze()
        #img = test[i, :, :].squeeze()
        #trg = test_targets[i, :, :].squeeze()

        # Adds color bar to uncertainty map and saves.
        #fig, ax = plt.subplots()
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #im = ax.imshow(unc_slice, cmap=cc.cm.CET_L19)
        #fig.colorbar(im, cax=cax, orientation="vertical")
        #plt.savefig(images_dir + prefix + "unc_{}.png".format(i))
        #plt.close()

        # Saves image plots.
        #plt.imsave(images_dir + prefix + "pred_{}.png".format(i),
        #           pred_slice, cmap="Greys")
        #plt.imsave(images_dir + "/img_{}.png".format(i), img, cmap="Greys")
        #plt.imsave(images_dir + "/target_{}.png".format(i), trg, cmap="Greys")


@ex.capture
def predict(model, test, test_targets, test_coords, test_shape,
            input_shape, vnet, bayesian, batch_size, border_trim, mc_samples):
    """Uses given model to predict on test data."""

    # Initializes prediction variables.
    sigmoids = None

    if vnet:
        # Initializes V-Net specific prediction variables.
        sigmoids = np.zeros((mc_samples,) + test_shape)

        for i in range(mc_samples):
            sigmoid = np.zeros(test_shape)
            counts = np.zeros(test_shape)
            print("MC Sample {}/{}".format(i+1, mc_samples))

            # Predicts on individual chunks.
            for j, (chunk, coords) in enumerate(zip(test, test_coords)):

                # Performs Monte Carlo sampling.
                chunk_pred = model.predict(np.expand_dims(chunk, axis=0))[0]

                # Discards poor edge predictions.
                trimmed_shape = input_shape
                border1 = ceil(input_shape[0] * border_trim)
                border2 = ceil(input_shape[1] * border_trim)
                border3 = ceil(input_shape[2] * border_trim)

                # Checks edge cases on edge discarding.
                # For example, we don't want to throw away an edge
                # if it is the very edge of the volume, because that
                # edge may only get predicted on once.
                if coords[0] != 0 and coords[0] != test_shape[0] - input_shape[0]:
                    chunk_pred = chunk_pred[border1:-border1, :, :, :]
                    coords = [coords[0] + border1, coords[1], coords[2]]
                    trimmed_shape = [trimmed_shape[0] - (2 * border1), trimmed_shape[1], trimmed_shape[2], 1]
                elif coords[0] != 0:
                    chunk_pred = chunk_pred[border1:, :, :, :]
                    coords = [coords[0] + border1, coords[1], coords[2]]
                    trimmed_shape = [trimmed_shape[0] - border1, trimmed_shape[1], trimmed_shape[2], 1]
                elif coords[0] != test_shape[0] - input_shape[0]:
                    chunk_pred = chunk_pred[:-border1, :, :, :]
                    trimmed_shape = [trimmed_shape[0] - border1, trimmed_shape[1], trimmed_shape[2], 1]

                if coords[1] != 0 and coords[1] != test_shape[1] - input_shape[1]:
                    chunk_pred = chunk_pred[:, border2:-border2, :, :]
                    coords = [coords[0], coords[1] + border2, coords[2]]
                    trimmed_shape = [trimmed_shape[0], trimmed_shape[1] - (2 * border2), trimmed_shape[2], 1]
                elif coords[1] != 0:
                    chunk_pred = chunk_pred[:, border2:, :, :]
                    coords = [coords[0], coords[1] + border2, coords[2]]
                    trimmed_shape = [trimmed_shape[0], trimmed_shape[1] - border2, trimmed_shape[2], 1]
                elif coords[1] != test_shape[1] - input_shape[1]:
                    chunk_pred = chunk_pred[:, :-border2, :, :]
                    trimmed_shape = [trimmed_shape[0], trimmed_shape[1] - border2, trimmed_shape[2], 1]

                if coords[2] != 0 and coords[2] != test_shape[2] - input_shape[2]:
                    chunk_pred = chunk_pred[:, :, border3:-border3, :]
                    coords = [coords[0], coords[1], coords[2] + border3]
                    trimmed_shape = [trimmed_shape[0], trimmed_shape[1], trimmed_shape[2] - (2 * border3), 1]
                elif coords[2] != 0:
                    chunk_pred = chunk_pred[:, :, border3:, :]
                    coords = [coords[0], coords[1], coords[2] + border3]
                    trimmed_shape = [trimmed_shape[0], trimmed_shape[1], trimmed_shape[2] - border3, 1]
                elif coords[2] != test_shape[2] - input_shape[2]:
                    chunk_pred = chunk_pred[:, :, :-border3, :]
                    trimmed_shape = [trimmed_shape[0], trimmed_shape[1], trimmed_shape[2] - border3, 1]

                # Increments each voxel in the counts array.
                counts = add_chunk_to_arr(counts, np.ones(trimmed_shape),
                                          coords, trimmed_shape)

                # Updates the sigmoid volume with the voxel means.
                sigmoid = add_chunk_to_arr(sigmoid, chunk_pred, coords, trimmed_shape)

            # Divides each voxel by the number of times it was predicted.
            sigmoid = sigmoid / counts
            sigmoids[i] = sigmoid

    else:
        # Predicts on entire slices.
        print()
        sigmoids = np.zeros((mc_samples,) + test_shape)

        # Performs Monte Carlo sampling.
        for i in range(mc_samples):
            sigmoids[i] = model.predict(test, batch_size=batch_size)

    # Calculates prediction.
    pred = np.mean(sigmoids, axis=0)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.

    # If data was chunked, turn it back into the original size.
    if vnet and test_coords is not None and test_shape is not None:
        test = reconstruct(test, test_coords, test_shape)
        test_targets = reconstruct(test_targets, test_coords, test_shape)

    # Saves predictions.
    save_predictions(sigmoids, pred, test, test_targets)


@ex.automain
def test(weights_path, batch_size):
    """Tests a model."""

    try:
        # Loads or creates test data.
        input_shape, test, test_targets, \
            test_coords, orig_test_shape = get_test_data()
    except FileNotFoundError as e:
        print(e)
        print("Could not find test files in data_dir. "
              "Did you specify the correct orig_test_data_dir?")
        return

    # Loads or creates model.
    model, checkpoint_path, _ = get_model(input_shape,
                                       scale_factor=len(test)/batch_size,
                                       weights_path=weights_path)

    # Predicts on test data and saves results.
    predict(model, test, test_targets, test_coords,
            orig_test_shape, input_shape)
    plots()
