def make_val_step_fn(model, loss_fn):
    def perform_val_step_fn(x, y):
        model.eval()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        return loss.item()
    return perform_val_step_fn
