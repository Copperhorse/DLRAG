import numpy as np

def make_data():
    true_b = 1
    true_w = 2
    N = 100
    np.random.seed(42)
    x = np.random.rand(N, 1)
    y = true_b + true_w * x + (.1 * np.random.randn(N, 1))
    idx = np.arange(N)
    np.random.shuffle(idx)
    train_idx = idx[:int(N*.8)]
    val_idx = idx[int(N*.8):]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    return x_train, y_train, x_val, y_val
