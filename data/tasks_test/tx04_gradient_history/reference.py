def gradient_history(sbs, layer_name, n_epochs):
    sbs.capture_gradients([layer_name])
    sbs.train(n_epochs)
    history = sbs._gradients[layer_name]
    sbs.remove_hooks()
    return history
