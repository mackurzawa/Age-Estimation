from Constants import AGE_THRESHOLDS, BATCH_SIZE, EPOCHS
from Preprocess import calculate_class_weights
from AgeImageDataset import AgeImageDataset
from Model import Model
from Training import train_model
from Saving import save_model

import os
import torch
import mlflow
from torch.utils.data import DataLoader

class_weights = calculate_class_weights()

train_dataset = AgeImageDataset(os.path.join('..', 'Data', 'Train'))
test_dataset = AgeImageDataset(os.path.join('..', 'Data', 'Test'))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device used for training:', device)

model = Model(len(AGE_THRESHOLDS) - 1)

with mlflow.start_run():
    model, train_losses, test_losses, train_acc, test_acc = train_model(model, train_dataloader, test_dataloader, device, class_weights)
    mlflow.log_param("Epoch", EPOCHS)
    mlflow.log_param("Batch size", BATCH_SIZE)
print('Training done!')

save_model(model)
print('Saving done!')
