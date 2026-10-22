def weight_history(sbs, layer_name, n_epochs):
    sbs.capture_parameters([layer_name])
    sbs.train(n_epochs)
    history = sbs._parameters[layer_name]
    sbs.remove_hooks()
    return history
