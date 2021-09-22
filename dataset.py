import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import glob
from PIL import Image


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, mode, transform=None):
        self.all_data = sorted(glob.glob(os.path.join(data_dir, mode, '*')))
        self.transform = transform

    def __len__(self):
        length = len(self.all_data)
        return length

    def __getitem__(self, idx):
        data_path = self.all_data[idx]
        img = Image.open(data_path)
        if self.transform is not None:
            img = self.transform(img)

        basename = os.path.basename(data_path)
        for lbl in range(0,10):
            if lbl == int(basename[6]):
                label = lbl
                label = torch.tensor(label)
                break
            else :
                continue

        return img, label

if __name__ == '__main__':

    dir = "./data"

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    trainset = MNIST(data_dir=dir, mode='train', transform=transform)
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

    for img, label in train_loader:
        print(img)
        print(label)

        import sys;
        sys.exit(0)

