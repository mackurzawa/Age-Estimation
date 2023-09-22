from Constants import AGE_THRESHOLDS
from Parameters import BATCH_SIZE, EPOCHS, EARLY_STOP_CALLBACK_PATIENCE, MODEL_TYPE
from Preprocess import calculate_class_weights
from AgeImageDataset import AgeImageDataset
from Model import SimpleModel, create_resnet, create_densenet
from Training import train_model
from Saving import save_model

import os
import torch
import mlflow
from torch.utils.data import DataLoader
import torch.nn as nn

class_weights = calculate_class_weights()

train_dataset = AgeImageDataset(os.path.join('..', 'Data', 'Train'), MODEL_TYPE)
test_dataset = AgeImageDataset(os.path.join('..', 'Data', 'Test'), MODEL_TYPE)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device used for training:', device)

if MODEL_TYPE == 'Simple':
    model = SimpleModel(len(AGE_THRESHOLDS) - 1)
elif MODEL_TYPE == 'ResNet50':
    model = create_resnet(len(AGE_THRESHOLDS) - 1)
elif MODEL_TYPE == 'DenseNet161':
    model = create_densenet(len(AGE_THRESHOLDS) - 1)
else:
    raise ValueError('Wrong MODEL_TYPE !!!')

with mlflow.start_run():
    model, train_losses, test_losses, train_acc, test_acc = train_model(model, MODEL_TYPE, train_dataloader, test_dataloader, device, class_weights, EPOCHS, EARLY_STOP_CALLBACK_PATIENCE)
    mlflow.log_param("Epoch", EPOCHS)
    mlflow.log_param("Batch size", BATCH_SIZE)
print('Training done!')

save_model(model, MODEL_TYPE)
