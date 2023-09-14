from Constants import EPOCHS
from Callbacks import calculate_accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow
from tqdm import tqdm


def train_model(model, train_dataloader, test_dataloader, device, class_weights):
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    print('class weights', class_weights)
    # loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

    train_losses = []
    test_losses = []

    train_acc = []
    test_acc = []

    for epoch in range(EPOCHS):
        train_losses_epoch = []
        test_losses_epoch = []

        train_acc_epoch = []
        test_acc_epoch = []

        # Train
        for X, y in tqdm(train_dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(X)
            pred = pred.to(torch.float32)

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_losses_epoch.append(loss.item())
            acc = calculate_accuracy(pred, y)
            train_acc_epoch.append(acc)

        # Test
        for X, y in tqdm(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.to(torch.float32)
            loss = loss_fn(pred, y)
            test_losses_epoch.append(loss.item())

            acc = calculate_accuracy(pred, y)
            test_acc_epoch.append(acc)

        train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))
        test_losses.append(sum(test_losses_epoch) / len(test_losses_epoch))
        train_acc.append(sum(train_acc_epoch) / len(train_acc_epoch))
        test_acc.append(sum(test_acc_epoch) / len(test_acc_epoch))

        print(f'Epoch {epoch + 1} completed')
        print(f'Train loss: {train_losses[-1]}')
        print(f'Test loss: {test_losses[-1]}')
        print(f'Train accuracy: {train_acc[-1]}')
        print(f'Test accuracy: {test_acc[-1]}')
        print()
        mlflow.log_metric("train_losses", train_losses[-1])
        mlflow.log_metric("test_losses", test_losses[-1])
        mlflow.log_metric("train_acc", train_acc[-1])
        mlflow.log_metric("test_acc", test_acc[-1])

    return model, train_losses, test_losses, train_acc, test_acc
