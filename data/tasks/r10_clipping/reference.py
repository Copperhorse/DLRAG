import torch.nn as nn

class StepByStep:
    def set_clip_grad_value(self, clip_value):
        self.clipping = lambda: nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_value)

    def set_clip_grad_norm(self, max_norm, norm_type=2):
        self.clipping = lambda: nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=norm_type)

    def remove_clip(self):
        self.clipping = None
