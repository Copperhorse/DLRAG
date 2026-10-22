class StepByStep:
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
