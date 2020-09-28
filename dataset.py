import torch.utils.data as data
import torchvision
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('L')
    y = img
    randint = np.random.randint(0, 3)
    if randint == 0:
        y = y.rotate(90)
    if randint == 1:
        y = y.rotate(180)
    if randint == 2:
        y = y.rotate(270)
    else:
        pass

    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, other_dir, LR_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.filenames = [join(other_dir, x) for x in listdir(other_dir) if is_image_file(x)]
        self.LR_transform = LR_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        put = load_img(self.filenames[index])
        # HR_4 = self.HR_4_transform(HR_4)
        LR = self.LR_transform(input)
        HR = self.LR_transform(put)
        return LR, HR

    def __len__(self):
        return len(self.image_filenames)
