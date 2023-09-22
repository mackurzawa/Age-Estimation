import os
from datetime import datetime
import torch


def save_model(model, model_type):
    now = datetime.now()
    model_name = f'{now.year}.{now.month}.{now.day} {now.hour};{now.minute}'
    print(f"Model saved as '{model_name}'")
    model_path = os.path.join('Models', f'{model_type}', f'Model-{model_name}')
    torch.save(model.state_dict(), model_path)
