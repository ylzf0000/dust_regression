import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv

real_path = os.path.realpath(__file__)
real_dir = real_path[:real_path.rfind('/')]
data_dir = os.path.join(real_dir, 'data')
print(data_dir)

transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class DustDataset(Dataset):
    def __init__(self, data_path):
        self.images = glob.glob(os.path.join(data_path, '*.jpg'))
        self.len = len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        # x = cv.imread(img)
        x = Image.open(img)
        x = transform(x)
        y = int(img[img.rfind('c') + 1:-4])
        # y = F.one_hot(torch.as_tensor([y]), num_classes=3)[0]
        return x, y

    def __len__(self):
        return self.len


# def my_collate(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     return [data, target]

dust_train_dataset = DustDataset('data_train')
dust_valid_dataset = DustDataset('data_valid')
dust_test_dataset = DustDataset('data_test')

dust_train_dataloader = DataLoader(
    dust_train_dataset, batch_size=64, shuffle=True)
dust_valid_dataloader = DataLoader(
    dust_valid_dataset, batch_size=32, shuffle=True)
dust_test_dataloader = DataLoader(
    dust_test_dataset, batch_size=32, shuffle=True)

if __name__ == '__main__':
    for X, Y in dust_train_dataloader:
        print(X.shape, Y.shape)
