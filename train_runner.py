from src.train import Trainer

def main():
    trainer = Trainer(data_dir="dataset", epochs=10)
    trainer.train()

if __name__ == "__main__":
    main()