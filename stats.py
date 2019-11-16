import csv
import os

from brokenaxes import brokenaxes
import matplotlib.pyplot as plt
import numpy as np

from utils import acc, ex

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@ex.capture
def stats(predict_dir):
    """Calculates prediction and uncertainty statistics."""

    bp = np.load(predict_dir + "/bayesian/bayesian_pred.npy").squeeze()
    bu = np.load(predict_dir + "/bayesian/bayesian_unc.npy").squeeze()
    dp = np.load(predict_dir + "/dropout/dropout_pred.npy").squeeze()
    du = np.load(predict_dir + "/dropout/dropout_unc.npy").squeeze()
    y = np.load(predict_dir + "/test_targets.npy").squeeze()

    with open(predict_dir + "/stats.csv", "w") as csvfile:
        w = csv.writer(csvfile, delimiter=" ")
        w.writerow(["Category", "Dropout", "Bayesian"])
        w.writerow(["Pred_Acc", acc(dp, y), acc(bp, y)])
        w.writerow(["Unc_Mean", du.mean(), bu.mean()])
        w.writerow(["Unc_Var", du.var(), bu.var()])
        w.writerow(["Unc_Max", du.max(), bu.max()])
        w.writerow(["Unc_Min", du.min(), bu.min()])


@ex.capture
def plots(images_dir, predict_dir):
    """Plots histograms of uncertainty values."""

    bu = np.load(predict_dir + "/bayesian/bayesian_unc.npy").flatten()
    du = np.load(predict_dir + "/dropout/dropout_unc.npy").flatten()

    # Removes extreme outliers so plot isn't stretched out.
    xlim = round(max(np.percentile(bu, 99.95), np.percentile(du, 99.95)), 2)
    bu = bu[bu < xlim]
    du = du[du < xlim]

    # Automatically calculates y-axis heights.
    bu_max = np.count_nonzero(bu == 0.)
    bu_mid = np.partition(np.histogram(bu, bins=50)[0], -2)[-2]
    du_max = np.count_nonzero(du == 0.)
    du_mid = np.partition(np.histogram(du, bins=50)[0], -2)[-2]

    # Plots histogram of Bayesian uncertainty map.
    fig = plt.figure()
    if bu_mid > 0:
        bax = brokenaxes(ylims=((0, bu_mid), (bu_max - (bu_mid / 5), bu_max)))
        bax.hist(bu, bins=50)
    else:
        plt.hist(bu, bins=50)

    plt.title("Distribution of Bayesian uncertainty map")
    # plt.xlabel("Uncertainty value")
    # plt.ylabel("Count")
    plt.savefig(images_dir + "/bayesian/bayesian_unc_dist.png")
    plt.clf()

    # Plots histogram of dropout uncertainty map.
    fig = plt.figure()
    if du_mid > 0:
        bax = brokenaxes(ylims=((0, du_mid), (du_max - (du_mid / 5), du_max)))
        bax.hist(du, bins=50)
    else:
        plt.hist(du, bins=50)

    plt.title("Distribution of dropout uncertainty map")
    # plt.xlabel("Uncertainty value")
    # plt.ylabel("Count")
    plt.savefig(images_dir + "/dropout/dropout_unc_dist.png")
    plt.clf()


@ex.automain
def get_stats_and_plots(images_dir):
    os.makedirs(images_dir + "/bayesian", exist_ok=True)
    os.makedirs(images_dir + "/dropout", exist_ok=True)

    stats()
    plots()
