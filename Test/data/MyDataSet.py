import os
import torch
import cv2
import random
from torch.utils.data import Dataset

# data path is a txt file which constains all the data's path
class MyDataSet(Dataset):
    def __init__(self, rootFolder, dataListPath):
        self.rootFolder = rootFolder
        # Load datalist
        with open(dataListPath) as f:
            self.datalist = [line.rstrip('\n') for line in f]

    # argument operation
    def argument(self, image, flipCode):
        return cv2.flip(image, flipCode)

    def __getitem__(self, index):
        image_path = self.datalist[index]
        label_path = image_path.replace('images', 'labels')
        image_path = os.path.join(self.rootFolder, image_path)
        label_path = os.path.join(self.rootFolder, label_path)
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)        

        # change rgb to gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # change channels from 3 to 1
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255

        # argument operation
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.argument(image, flipCode)
            label = self.argument(label, flipCode)

        return image, label

    def __len__(self):
        return len(self.datalist)