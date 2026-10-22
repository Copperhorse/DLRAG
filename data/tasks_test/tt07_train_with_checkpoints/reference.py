class StepByStep:
    def train_with_checkpoints(self, n_epochs, every, template, seed=42):
        done = 0
        while done < n_epochs:
            chunk = min(every, n_epochs - done)
            self.train(chunk, seed=seed)
            done += chunk
            self.save_checkpoint(template.format(epoch=self.total_epochs))
