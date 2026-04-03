from src.preprocessing import BasicPreprocessing
import matplotlib.pyplot as plt


def main():
    # 👉 Change this if your dataset path is different
    data_path = "dataset"

    prep = BasicPreprocessing(data_path)

    # Load dataset
    image_paths, labels = prep.import_dataset()

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(
        image_paths, labels
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = prep.create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # 🔍 Visualize one sample (VERY IMPORTANT)
    images, labels = next(iter(train_loader))

    plt.imshow(images[0].permute(1, 2, 0))
    plt.title(f"Label: {labels[0].item()}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()