from stepbystep.v4 import StepByStep

def image_normalizer(loader):
    return StepByStep.make_normalizer(loader)
