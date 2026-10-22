import torch

def layer_output_shapes(sbs, loader, layer_names):
    sbs.attach_hooks(layer_names)
    for x, y in loader:
        sbs.predict(x)
    shapes = {name: tuple(sbs.visualization[name].shape) for name in layer_names}
    sbs.remove_hooks()
    return shapes
