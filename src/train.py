import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.model import MaskClassifier
from src.preprocessing import BasicPreprocessing


class Trainer:
    def __init__(self, data_dir, epochs=10, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MaskClassifier().to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.epochs = epochs

        # Load data
        prep = BasicPreprocessing(data_dir)
        image_paths, labels = prep.import_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(image_paths, labels)

        self.train_loader, self.val_loader, self.test_loader = prep.create_dataloaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        return total_loss / len(self.train_loader), acc

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        return total_loss / len(self.val_loader), acc

    def train(self):
        best_val_acc = 0

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_acc.append(train_acc)
            self.val_acc.append(val_acc)

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "models/best_model.pth")
                print("✅ Best model saved!")

        self.plot_results()

    def plot_results(self):
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.legend()
        plt.savefig("results/training_curves.png")
        plt.close()