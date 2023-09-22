import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.l1 = nn.Conv2d(3, 8, 5)
        self.l2 = nn.Conv2d(8, 64, 5)
        self.l3 = nn.Conv2d(64, 128, 5)
        self.l4 = nn.Conv2d(128, 256, 5)
        self.l5 = nn.Conv2d(256, 256, 5)
        self.l6 = nn.Linear(1024, 128)
        self.l7 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.l5(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.l6(x))
        x = self.l7(x)

        return x.double()


def create_resnet(class_number):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model = nn.Sequential(
        model,
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, class_number),
        nn.Softmax()
    )
    return model


def create_densenet(class_number):
    model = models.densenet161(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model = nn.Sequential(
        model,
        nn.Linear(1000, 512),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(128, class_number),
        nn.Dropout(0.25),
        nn.Softmax()
    )
    return model
