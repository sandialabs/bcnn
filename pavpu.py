import numpy as np

from utils import ex

@ex.capture()
def calculate_pavpu(prediction, label, uncertainty,
                    accuracy_threshold=0.5, uncertainty_threshold=0.2, window_size=3):
    accurate_certain = 0.
    inaccurate_certain = 0.
    accurate_uncertain = 0.
    inaccurate_uncertain = 0.

    anchor = (0, 0)
    last_anchor = (prediction.shape[0] - window_size, prediction.shape[1] - window_size)

    while anchor != last_anchor:
        prediction_window = np.array([prediction[i][j]
                                      for i in range(anchor[0], anchor[0] + window_size)
                                      for j in range(anchor[1], anchor[1] + window_size)])

        label_window = np.array([label[i][j]
                                 for i in range(anchor[0], anchor[0] + window_size)
                                 for j in range(anchor[1], anchor[1] + window_size)])

        uncertainty_window = np.array([uncertainty[i][j]
                                       for i in range(anchor[0], anchor[0] + window_size)
                                       for j in range(anchor[1], anchor[1] + window_size)])

        accuracy = np.sum(prediction_window == label_window) / (window_size ** 2)
        avg_uncertainty = uncertainty_window.mean()

        accurate = True if accuracy >= accuracy_threshold else False
        uncertain = True if avg_uncertainty >= uncertainty_threshold else False

        if accurate:
            if uncertain:
                accurate_uncertain += 1
            else:
                accurate_certain += 1
        else:
            if uncertain:
                inaccurate_uncertain += 1
            else:
                inaccurate_certain += 1

        if anchor[1] < prediction.shape[1] - window_size:
            anchor = (anchor[0], anchor[1] + 1)
        else:
            anchor = (anchor[0] + 1, 0)

    print("AC: {}".format(accurate_certain))
    print("AU: {}".format(accurate_uncertain))
    print("IC: {}".format(inaccurate_certain))
    print("IU: {}".format(inaccurate_uncertain))

    a_given_c = accurate_certain / (accurate_certain + inaccurate_certain)
    u_given_i = inaccurate_uncertain / (inaccurate_certain + inaccurate_uncertain)

    print("A|C: {}".format(a_given_c))
    print("U|I: {}".format(u_given_i))

    pavpu = (accurate_certain + inaccurate_uncertain) / (accurate_certain + accurate_uncertain + inaccurate_certain + inaccurate_uncertain)

    print("PAvPU: {}".format(pavpu))


@ex.automain
def main():
    label = np.load("/data/wg-cee-dev-dgx/output/bcnn/predict/graphite/Litarion/test_targets.npy")[324]
    pred = np.load("/data/wg-cee-dev-dgx/output/bcnn/predict/graphite/Litarion/dropout/dropout_pred.npy")[324]
    std = np.load("/data/wg-cee-dev-dgx/output/bcnn/predict/graphite/Litarion/dropout/dropout_unc.npy")[324]

    val = std.mean()# + (std.std())
    calculate_pavpu(pred, label, std, accuracy_threshold=0.78, uncertainty_threshold=val)
    print() 

    pred = np.load("/data/wg-cee-dev-dgx/output/bcnn/predict/graphite/Litarion/bayesian/bayesian_pred.npy")[324]
    std = np.load("/data/wg-cee-dev-dgx/output/bcnn/predict/graphite/Litarion/bayesian/bayesian_unc.npy")[324]

    val = std.mean()# + (std.std())
    calculate_pavpu(pred, label, std, accuracy_threshold=0.78, uncertainty_threshold=val)

