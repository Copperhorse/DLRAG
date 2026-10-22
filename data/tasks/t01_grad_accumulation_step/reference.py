def make_accum_train_step_fn(model, loss_fn, optimizer, accum_steps):
    counter = {'n': 0}
    def perform_train_step_fn(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        (loss / accum_steps).backward()
        counter['n'] += 1
        if counter['n'] % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    return perform_train_step_fn
