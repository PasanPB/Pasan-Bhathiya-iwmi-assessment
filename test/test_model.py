import torch
from src.model import MaskClassifier

def main():
    model = MaskClassifier()

    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)

    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()