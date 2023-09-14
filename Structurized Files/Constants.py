import os

AGE_THRESHOLDS = [0, 15, 30, 45, 60, 116]
FILENAMES = os.listdir(os.path.join('..', 'Data', 'UTKFace'))

BATCH_SIZE = 32
EPOCHS = 2
