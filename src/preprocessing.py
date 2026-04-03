import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class BasicPreprocessing:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def import_dataset(self):
        image_paths = []
        labels = []

        categories = ["with_mask", "without_mask"]

        for label, category in enumerate(categories):
            folder_path = os.path.join(self.data_dir, category)

            if not os.path.exists(folder_path):
                raise Exception(f"Folder not found: {folder_path}")

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(file_path)
                    labels.append(label)

        print(f"Total images loaded: {len(image_paths)}")
        return image_paths, labels

    def split_data(self, image_paths, labels):
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels, test_size=0.3, random_state=42, stratify=labels
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        return train_transform, test_transform

    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        train_transform, test_transform = self.get_transforms()

        train_dataset = CustomDataset(X_train, y_train, train_transform)
        val_dataset = CustomDataset(X_val, y_val, test_transform)
        test_dataset = CustomDataset(X_test, y_test, test_transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader, test_loader