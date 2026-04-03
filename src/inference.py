import torch
import cv2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import ModelDevelopment


class BasicInference:

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = ModelDevelopment().get_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load Haarcascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.classes = ["with_mask", "without_mask"]

    def preprocess_face(self, face_img):
        face = cv2.resize(face_img, (128, 128))
        face = face / 255.0
        face = np.transpose(face, (2, 0, 1))  # HWC → CHW
        face = np.expand_dims(face, axis=0)
        face_tensor = torch.tensor(face, dtype=torch.float32).to(self.device)
        return face_tensor

    def detect_images(self, image_path):
        """
        Detect faces and classify mask / no mask
        """

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        results = []

        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]

            input_tensor = self.preprocess_face(face)

            with torch.no_grad():
                output = self.model(input_tensor)
                _, pred = torch.max(output, 1)

            label = self.classes[pred.item()]
            results.append(label)

            # Draw bounding box
            color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return results

    def evaluate_model(self, data_loader):
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=self.classes,
                    yticklabels=self.classes)
        plt.title("Confusion Matrix")
        plt.savefig("results/confusion_matrix.png")
        plt.close()

    def any_name(self):
        print("Custom inference method placeholder")


# 🔹 Main function (as required)
def main():
    print("IWMI Data Science Internship Assessment, I’m not a data scientist")

    inference = BasicInference("models/best_model.pth")

    # Test with one image
    test_image = "dataset/with_mask/sample.jpg"  # change this
    inference.detect_images(test_image)


if __name__ == "__main__":
    main()