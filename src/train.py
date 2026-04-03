import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from src.model import ModelDevelopment
from src.preprocessing import BasicPreprocessing


class Trainer:

    def __init__(self, data_dir, epochs=10, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🔹 Model (from required structure)
        self.model = ModelDevelopment().get_model().to(self.device)

        # 🔹 Loss & Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 🔹 Learning Rate Scheduler (REQUIRED)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.epochs = epochs

        # 🔹 Load Data
        prep = BasicPreprocessing(data_dir)
        df = prep.import_dataset()
        X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(df)

        self.train_loader, self.val_loader, self.test_loader = prep.create_dataloaders(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )

        # 🔹 Tracking Metrics
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

        # Ensure folders exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    # 🔹 Train One Epoch
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

        accuracy = correct / total
        return total_loss / len(self.train_loader), accuracy

    # 🔹 Validation
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

        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy

    # 🔹 Training Loop
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

            print(f"\nEpoch [{epoch+1}/{self.epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 🔹 Save Best Model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "models/best_model.pth")
                print("✅ Best model saved!")

        # After training
        self.plot_results()

    # 🔹 Plot Results
    def plot_results(self):
        plt.figure(figsize=(10, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.title("Loss Curve")
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label="Train Acc")
        plt.plot(self.val_acc, label="Val Acc")
        plt.title("Accuracy Curve")
        plt.legend()

        plt.savefig("results/training_curves.png")
        plt.close()