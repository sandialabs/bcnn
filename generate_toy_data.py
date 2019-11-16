import os

import numpy as np

def generate_toy_data():
    os.makedirs("toy_data/orig/train", exist_ok=True)
    os.makedirs("toy_data/orig/valid", exist_ok=True)
    os.makedirs("toy_data/orig/test", exist_ok=True)
    os.makedirs("toy_data/orig/train_targets", exist_ok=True)
    os.makedirs("toy_data/orig/valid_targets", exist_ok=True)
    os.makedirs("toy_data/orig/test_targets", exist_ok=True)

    a = np.full((176, 600, 600, 1), 0.1)
    a[:, 100:500, 100:500, :] = 0.9
    np.save("toy_data/orig/train/train.npy", a)
    np.save("toy_data/orig/valid/valid.npy", a[:88])
    del a

    b = np.full((88, 600, 600, 1), 0.1)
    b[:, 100:500, 100:500, :] = 0.9
    b[:, 50:100, 50:550, :] = 0.6
    b[:, 500:550, 50:550, :] = 0.6
    b[:, 50:550, 50:100, :] = 0.6
    b[:, 50:550, 500:550, :] = 0.6
    np.save("toy_data/orig/test/test.npy", b)
    del b

    c = np.zeros((176, 600, 600, 1))
    c[:, 100:500, 100:500, :] = 1.
    np.save("toy_data/orig/train_targets/train_targets.npy", c)
    np.save("toy_data/orig/valid_targets/valid_targets.npy", c[:88])
    c[:, 50:100, 50:550, :] = 1.
    c[:, 500:550, 50:550, :] = 1.
    c[:, 50:550, 50:100, :] = 1.
    c[:, 50:550, 500:550, :] = 1.
    np.save("toy_data/orig/test_targets/test_targets.npy", c[:88])
    del c

if __name__ == "__main__":
    generate_toy_data()
