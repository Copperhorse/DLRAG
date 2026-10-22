import numpy as np
import matplotlib.pyplot as plt

def suggest_lr(sbs, loader, end_lr=1.0, num_iter=30):
    tracking, fig = sbs.lr_range_test(loader, end_lr=end_lr, num_iter=num_iter)
    plt.close(fig)
    best = int(np.argmin(tracking['loss']))
    return float(tracking['lr'][best])
