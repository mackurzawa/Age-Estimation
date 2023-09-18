import os

AGE_THRESHOLDS = [0, 15, 30, 45, 60, 116]
FILENAMES = os.listdir(os.path.join('..', 'Data', 'UTKFace'))

EARLY_STOP_CALLBACK_PATIENCE = 15

BATCH_SIZE = 32
EPOCHS = 100
