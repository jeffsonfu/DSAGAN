from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, RandomCrop, RandomHorizontalFlip

from dataset import DatasetFromFolder
crop_size = 128


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)

    return output_image_dir


def LR_transform(crop_size):
    return Compose([
        Scale(crop_size//4),
        ToTensor(),
    ])


def get_training_set():
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    other_dir = join(root_dir, "other")
    return DatasetFromFolder(train_dir, other_dir,
                             LR_transform=LR_transform(crop_size))


def get_test_set():
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    other_dir = join(root_dir, "other")
    return DatasetFromFolder(test_dir, other_dir,
                             LR_transform=LR_transform(crop_size))
