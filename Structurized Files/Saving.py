import os
from datetime import datetime
import torch


def save_model(model):
    now = datetime.now()
    model_path = os.path.join('Models', f'Model-{now.year}.{now.month}.{now.day} {now.hour};{now.minute}')
    torch.save(model.state_dict(), model_path)