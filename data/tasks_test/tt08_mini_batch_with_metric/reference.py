import numpy as np

def mini_batch_with_metric(device, data_loader, step_fn):
    losses, metrics = [], []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        loss, metric = step_fn(x_batch, y_batch)
        losses.append(loss)
        metrics.append(metric)
    return np.mean(losses), np.mean(metrics)
