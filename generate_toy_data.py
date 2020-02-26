import os

import numpy as np

def generate_toy_data():
    os.makedirs("toy_data/orig/train", exist_ok=True)
    os.makedirs("toy_data/orig/valid", exist_ok=True)
    os.makedirs("toy_data/orig/test", exist_ok=True)
    os.makedirs("toy_data/orig/train_targets", exist_ok=True)
    os.makedirs("toy_data/orig/valid_targets", exist_ok=True)
    os.makedirs("toy_data/orig/test_targets", exist_ok=True)

    a = np.full((80, 120, 120, 1), 0.1)
    a[:, 20:100, 20:100, :] = 0.9
    np.save("toy_data/orig/train/train.npy", a)
    np.save("toy_data/orig/valid/valid.npy", a[:40])
    del a

    b = np.full((40, 120, 120, 1), 0.1)
    b[:, 20:100, 20:100, :] = 0.9
    b[:, 10:20, 10:110, :] = 0.7
    b[:, 100:110, 10:110, :] = 0.7
    b[:, 10:110, 10:20, :] = 0.7
    b[:, 10:110, 100:110, :] = 0.7
    np.save("toy_data/orig/test/test.npy", b)
    del b

    c = np.zeros((80, 120, 120, 1))
    c[:, 20:100, 20:100, :] = 1.
    np.save("toy_data/orig/train_targets/train_targets.npy", c)
    np.save("toy_data/orig/valid_targets/valid_targets.npy", c[:40])
    c[:, 10:20, 10:110, :] = 1.
    c[:, 100:110, 10:110, :] = 1.
    c[:, 10:110, 10:20, :] = 1.
    c[:, 10:110, 100:110, :] = 1.
    np.save("toy_data/orig/test_targets/test_targets.npy", c[:40])
    del c

if __name__ == "__main__":
    generate_toy_data()
