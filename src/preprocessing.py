import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class BasicPreprocessing:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def import_dataset(self):
        """
        Method
        ------
            Import the given dataset.

        Parameters
        ----------
            As required
        """

        image_paths = []
        labels = []

        categories = ["with_mask", "without_mask"]

        for label, category in enumerate(categories):
            folder_path = os.path.join(self.data_dir, category)

            for file_name in os.listdir(folder_path):
                if file_name.endswith(('.jpg', '.png', '.jpeg')):
                    full_path = os.path.join(folder_path, file_name)
                    image_paths.append(full_path)
                    labels.append(label)

        print(f"Total images: {len(image_paths)}")
        return image_paths, labels

    def split_data(self, data, labels=None):
        if labels is None:
            # Supports current training flow: split_data(df)
            X = data["image_path"].values
            y = data["label"].values
        else:
            # Supports test flow: split_data(image_paths, labels)
            X = np.array(data)
            y = np.array(labels)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        return train_transform, test_transform

    def create_dataloaders(self, X_train, X_val, X_test,
                           y_train, y_val, y_test):

        train_transform, test_transform = self.get_transforms()

        train_dataset = CustomDataset(X_train, y_train, train_transform)
        val_dataset = CustomDataset(X_val, y_val, test_transform)
        test_dataset = CustomDataset(X_test, y_test, test_transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader, test_loader

    # Additional helper method (good practice)
    def any_name(self):
        print("Custom preprocessing method placeholder")