import torch.nn as nn

def make_clipped_train_step_fn(model, loss_fn, optimizer, max_norm):
    def perform_train_step_fn(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return perform_train_step_fn
