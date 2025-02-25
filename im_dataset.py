import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from imageio.v2 import imread
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2


def crop_and_resize(img, w, h):
    im_h, im_w, channels = img.shape
    res_aspect_ratio = w/h
    input_aspect_ratio = im_w/im_h

    if input_aspect_ratio > res_aspect_ratio:
        im_w_r = int(input_aspect_ratio*h)
        im_h_r = h
        img = cv2.resize(img, (im_w_r , im_h_r))
        x1 = int((im_w_r - w)/2)
        x2 = x1 + w
        img = img[:, x1:x2, :]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w/input_aspect_ratio)
        img = cv2.resize(img, (im_w_r , im_h_r))
        y1 = int((im_h_r - h)/2)
        y2 = y1 + h
        img = img[y1:y2, :, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv2.resize(img, (w, h))

    return img
def get_datasets(**kwargs):

    to_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])

    train = ExerciseDataset(partition="train", transform=None, **kwargs)
    val = ExerciseDataset("validation", transform=None, **kwargs)
    test = ExerciseDataset("test", transform=None, **kwargs)

    """mean, std = train.X.mean(), train.X.std()
    norm = transforms.Normalize(mean, std)

    x_tensor = norm(torch.from_numpy(train.X.astype(float)))
    train.X = x_tensor.numpy()
    train.X.astype(int)"""

    standardizer = Standardize()
    standardizer.fit(train.X)

    train.X = standardizer.transform(train.X)
    val.X = standardizer.transform(val.X)
    test.X = standardizer.transform(test.X)

    train.X = train.X.transpose(0, 3, 1, 2)
    val.X = val.X.transpose(0, 3, 1, 2)
    test.X = test.X.transpose(0, 3, 1, 2)

    return train, val, test, standardizer

def get_loaders(batch_size, **kwargs):
    train, val, test = get_datasets(**kwargs)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class Standardize(object):
    
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Calculate per-channel mean and standard deviation from dataset X."""
        X_type = np.array(X, dtype = np.float64)
        self.mean = np.mean(X_type, axis = (0, 1, 2))
        self.std = np.std(X_type, axis = (0, 1, 2))
    
    def transform(self, X):
        """Return standardized dataset given dataset X."""
        N, image_height, image_width, color_channel = X.shape
        X_copy = np.array(X, dtype = np.float64)
        for i in range(N):
            for j in range(image_height):
                for k in range(image_width):
                    for c in range(color_channel):
                        X_copy[i][j][k][c] = (X_copy[i][j][k][c] - self.mean[c]) / self.std[c]
                    
        return X_copy

class ExerciseDataset(Dataset):
    def __init__(self, partition, transform = None):
        self.partition = partition
        self.metadata = pd.read_csv("archive/ims.csv")
        self.transform = transform
        self.X, self.y = self._load_data()

        self.exercise_labels = dict(zip(self.metadata["numeric_labels"], self.metadata["exercise_label"]))

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
    
    def __len__(self):
        return len(self.X)
    
    def _load_data(self):
        df = self.metadata[self.metadata.partition == self.partition]

        X = []
        y = []
        base_dir = "archive/"
        for i, im in df.iterrows():
            dir = im["exercise_label"]
            image = imread(os.path.join(base_dir+str(dir), im["image_files"]))
            image = crop_and_resize(image, 64, 64)
            if self.transform:
                image = self.transform(image)
            if image.shape == (64, 64, 3):
                X.append(image)
            y.append(im["numeric_labels"])
            
        return np.array(X), np.array(y)

if __name__ == "__main__":
    train, val, test, standardizer = get_datasets()
    print("Train:\t", len(train.X))
    print("Val:\t", len(val.X))
    print("Test:\t", len(test.X))
    print("Mean:\t", standardizer.mean)
    print("Std:\t", standardizer.std)
