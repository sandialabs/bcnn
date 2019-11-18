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
def plots(bayesian, predict_dir, images_dir, num_percentiles):
    prefix = "/bayesian/bayesian_" if bayesian else "/dropout/dropout_"

    sigmoid = np.load(predict_dir + prefix + "sigmoid.npy")
    pred = np.load(predict_dir + prefix + "pred.npy")
    percentiles = np.load(predict_dir + prefix + "percentiles.npy")
    unc = np.load(predict_dir + prefix + "unc.npy")
    test = np.load(predict_dir + "/test.npy")
    test_targets = np.load(predict_dir + "/test_targets.npy")

    # Change eventually -- this is just for plotting purposes.
    twenty = percentiles[num_percentiles // 5]
    eighty = percentiles[num_percentiles - ((num_percentiles // 5) + 1)]

    # Plots four slices from each output numpy array.
    four_slices = range(test.shape[0] // 5, test.shape[0], test.shape[0] // 5)
    for i in four_slices:
        sig_slice = sigmoid[i, :, :].squeeze()
        pred_slice = pred[i, :, :].squeeze()
        twenty_slice = twenty[i, :, :].squeeze()
        eighty_slice = eighty[i, :, :].squeeze()
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
        plt.imsave(images_dir + prefix + "sigmoid_{}.png".format(i),
                   sig_slice, cmap="Greys")
        plt.imsave(images_dir + prefix + "pred_{}.png".format(i),
                   pred_slice, cmap="Greys")
        plt.imsave(images_dir + prefix + "twenty_{}.png".format(i),
                   twenty_slice, cmap="Greys")
        plt.imsave(images_dir + prefix + "eighty_{}.png".format(i),
                   eighty_slice, cmap="Greys")
        plt.imsave(images_dir + "/img_{}.png".format(i), img, cmap="Greys")
        plt.imsave(images_dir + "/target_{}.png".format(i), trg, cmap="Greys")


@ex.capture
def save_predictions(sigmoid, pred, percentiles, unc, test, test_targets,
                     bayesian, predict_dir, images_dir, num_percentiles):
    """Saves results of predictions."""

    os.makedirs(predict_dir + "/bayesian", exist_ok=True)
    os.makedirs(predict_dir + "/dropout", exist_ok=True)
    os.makedirs(images_dir + "/bayesian", exist_ok=True)
    os.makedirs(images_dir + "/dropout", exist_ok=True)

    prefix = "/bayesian/bayesian_" if bayesian else "/dropout/dropout_"

    # Saves output numpy arrays.
    np.save(predict_dir + prefix + "sigmoid.npy", sigmoid)
    np.save(predict_dir + prefix + "pred.npy", pred)
    np.save(predict_dir + prefix + "percentiles.npy", percentiles)
    np.save(predict_dir + prefix + "unc.npy", unc)
    np.save(predict_dir + "/test.npy", test)
    np.save(predict_dir + "/test_targets.npy", test_targets)

    # Change eventually -- this is just for plotting purposes.
    twenty = percentiles[num_percentiles // 5]
    eighty = percentiles[num_percentiles - ((num_percentiles // 5) + 1)]

    # Plots four slices from each output numpy array.
    four_slices = range(test.shape[0] // 5, test.shape[0], test.shape[0] // 5)
    for i in four_slices:
        sig_slice = sigmoid[i, :, :].squeeze()
        pred_slice = pred[i, :, :].squeeze()
        twenty_slice = twenty[i, :, :].squeeze()
        eighty_slice = eighty[i, :, :].squeeze()
        unc_slice = unc[i, :, :].squeeze()
        trg = test_targets[i, :, :].squeeze()
        img = test[i, :, :].squeeze()

        # Adds color bar to uncertainty map and saves.
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(unc_slice, cmap="inferno")
        fig.colorbar(im, cax=cax, orientation="vertical")
        plt.savefig(images_dir + prefix + "unc_{}.png".format(i))
        plt.close()

        # Saves image plots.
        plt.imsave(images_dir + prefix + "sigmoid_{}.png".format(i),
                   sig_slice, cmap="Greys")
        plt.imsave(images_dir + prefix + "pred_{}.png".format(i),
                   pred_slice, cmap="Greys")
        plt.imsave(images_dir + prefix + "twenty_{}.png".format(i),
                   twenty_slice, cmap="Greys")
        plt.imsave(images_dir + prefix + "eighty_{}.png".format(i),
                   eighty_slice, cmap="Greys")
        plt.imsave(images_dir + "/img_{}.png".format(i), img, cmap="Greys")
        plt.imsave(images_dir + "/target_{}.png".format(i), trg, cmap="Greys")


@ex.capture
def predict(model, test, test_targets, test_coords, test_shape,
            input_shape, vnet, bayesian, batch_size,
            mc_samples, num_percentiles):
    """Uses given model to predict on test data."""

    # Ensures MC samples is divisible by batch size.
    if mc_samples < batch_size:
        raise ValueError("Not enough MC samples.")
    old_mc_samples = mc_samples
    mc_samples = round_down(old_mc_samples, batch_size)
    if old_mc_samples != mc_samples:
        print("MC samples rounded from {} to {}".format(old_mc_samples,
                                                        mc_samples))

    # Initializes prediction variables.
    sigmoid = None
    percentiles = None
    scale = 100 / (num_percentiles - 1)
    percentile_points = [scale*k for k in range(num_percentiles)]

    if vnet:
        # Initializes V-Net specific prediction variables.
        sigmoid = np.zeros(test_shape)
        counts = np.zeros(test_shape)
        percentiles = [np.zeros(test_shape) for i in range(num_percentiles)]

        # Predicts on individual chunks.
        print()
        for i, (chunk, coords) in enumerate(zip(test, test_coords)):
            print("Chunk {}/{}".format(i+1, test.shape[0]))
            chunk_samples = np.empty((mc_samples,) + input_shape)

            chunk = np.expand_dims(chunk, axis=0)
            batch = np.repeat(chunk, batch_size, axis=0)

            # Performs Monte Carlo sampling.
            for j in range(0, mc_samples, batch_size):
                chunk_samples[j:j+batch_size] = model.predict_on_batch(batch)

            # Discards poor edge predictions.
            # I use 5% but this can be changed.
            trimmed_shape = input_shape
            border1 = ceil(input_shape[0] * 0.05)
            border2 = ceil(input_shape[1] * 0.05)
            border3 = ceil(input_shape[2] * 0.05)

            # Checks edge cases on edge discarding.
            # For example, we don't want to throw away an edge
            # if it is the very edge of the volume, because that
            # edge may only get predicted on once.
            if coords[0] != 0 and coords[0] != test_shape[0] - input_shape[0]:
                chunk_samples = chunk_samples[:, border1:-border1, :, :, :]
                coords = [coords[0] + border1, coords[1], coords[2]]
                trimmed_shape = [trimmed_shape[0] - (2 * border1), trimmed_shape[1], trimmed_shape[2], 1]
            elif coords[0] != 0:
                chunk_samples = chunk_samples[:, border1:, :, :, :]
                coords = [coords[0] + border1, coords[1], coords[2]]
                trimmed_shape = [trimmed_shape[0] - border1, trimmed_shape[1], trimmed_shape[2], 1]
            elif coords[0] != test_shape[0] - input_shape[0]:
                chunk_samples = chunk_samples[:, :-border1, :, :, :]
                trimmed_shape = [trimmed_shape[0] - border1, trimmed_shape[1], trimmed_shape[2], 1]

            if coords[1] != 0 and coords[1] != test_shape[1] - input_shape[1]:
                chunk_samples = chunk_samples[:, :, border2:-border2, :, :]
                coords = [coords[0], coords[1] + border2, coords[2]]
                trimmed_shape = [trimmed_shape[0], trimmed_shape[1] - (2 * border2), trimmed_shape[2], 1]
            elif coords[1] != 0:
                chunk_samples = chunk_samples[:, :, border2:, :, :]
                coords = [coords[0], coords[1] + border2, coords[2]]
                trimmed_shape = [trimmed_shape[0], trimmed_shape[1] - border2, trimmed_shape[2], 1]
            elif coords[1] != test_shape[1] - input_shape[1]:
                chunk_samples = chunk_samples[:, :, :-border2, :, :]
                trimmed_shape = [trimmed_shape[0], trimmed_shape[1] - border2, trimmed_shape[2], 1]

            if coords[2] != 0 and coords[2] != test_shape[2] - input_shape[2]:
                chunk_samples = chunk_samples[:, :, :, border3:-border3, :]
                coords = [coords[0], coords[1], coords[2] + border3]
                trimmed_shape = [trimmed_shape[0], trimmed_shape[1], trimmed_shape[2] - (2 * border3), 1]
            elif coords[2] != 0:
                chunk_samples = chunk_samples[:, :, :, border3:, :]
                coords = [coords[0], coords[1], coords[2] + border3]
                trimmed_shape = [trimmed_shape[0], trimmed_shape[1], trimmed_shape[2] - border3, 1]
            elif coords[2] != test_shape[2] - input_shape[2]:
                chunk_samples = chunk_samples[:, :, :, :-border3, :]
                trimmed_shape = [trimmed_shape[0], trimmed_shape[1], trimmed_shape[2] - border3, 1]

            # Increments each voxel in the counts array.
            counts = add_chunk_to_arr(counts, np.ones(trimmed_shape),
                                      coords, trimmed_shape)

            # Updates the sigmoid volume with the voxel means.
            chunk_mean = np.mean(chunk_samples, axis=0)
            sigmoid = add_chunk_to_arr(sigmoid, chunk_mean, coords, trimmed_shape)

            # Updates the percentile volumes.
            percentile_samples = np.percentile(chunk_samples,
                                               percentile_points, axis=0)
            percentiles = [add_chunk_to_arr(p, s, coords, trimmed_shape)
                           for p, s in zip(percentiles, percentile_samples)]

        # Divides each voxel by the number of times it was predicted.
        sigmoid = sigmoid / counts

        # Note that division automatically broadcasts across axis 0.
        percentiles = percentiles / counts

    else:
        # Predicts on entire slices.
        print()
        samples = np.zeros((mc_samples,) + test_shape)

        # Performs Monte Carlo sampling.
        for i in range(mc_samples):
            samples[i] = model.predict(test, batch_size=batch_size)

        sigmoid = np.mean(samples, axis=0)
        percentiles = np.percentile(samples, percentile_points, axis=0)

    # Calculates prediction and uncertainty.
    pred = sigmoid.copy()
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.

    twenty = percentiles[num_percentiles // 5]
    eighty = percentiles[num_percentiles - ((num_percentiles // 5) + 1)]
    unc = eighty - twenty

    # If data was chunked, turn it back into the original size.
    if vnet and test_coords is not None and test_shape is not None:
        test = reconstruct(test, test_coords, test_shape)
        test_targets = reconstruct(test_targets, test_coords, test_shape)

    # Saves predictions.
    save_predictions(sigmoid, pred, percentiles, unc, test, test_targets)


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
    model, checkpoint_path = get_model(input_shape,
                                       scale_factor=len(test)/batch_size,
                                       weights_path=weights_path)

    # Predicts on test data and saves results.
    predict(model, test, test_targets, test_coords,
            orig_test_shape, input_shape)
    plots()
