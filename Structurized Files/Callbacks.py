import numpy as np
from copy import deepcopy


def calculate_accuracy(pred, y):
    pred = np.array(pred.cpu().detach().numpy())
    y = np.array(y.cpu())
    predicted_classes = [np.argmax(x) for x in pred]
    return sum(predicted_classes == y) / len(y)


# Early stop with restoring best weights !!!
class EarlyStopCallback:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model = None

    def early_stop(self, model, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = deepcopy(model)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True, self.best_model
        return False, model
