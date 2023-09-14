import numpy as np


def calculate_accuracy(pred, y):
    pred = np.array(pred.cpu().detach().numpy())
    y = np.array(y.cpu())
    predicted_classes = [np.argmax(x) for x in pred]
    return sum(predicted_classes == y)/len(y)