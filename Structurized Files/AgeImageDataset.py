from Constants import AGE_THRESHOLDS

import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class AgeImageDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.images_names = os.listdir(self.images_dir)

    def __len__(self):
        # return len(self.images_names)//10
        return len(self.images_names)

    def __getitem__(self, idx):
        image_name = self.images_names[idx]
        image = read_image(os.path.join(self.images_dir, image_name)) / 255.
        label_temp = image_name.split('_')[0]
        for i in range(len(AGE_THRESHOLDS) - 1):
            if AGE_THRESHOLDS[i] < int(label_temp) <= AGE_THRESHOLDS[i + 1]:
                label = i
                break
        return image.float(), label
