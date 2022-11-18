from data import load_data
import torch.optim as optim

def train


if __name__ == "__main__":
    ds = load_data("Headlines.db")
    train_ds = ds[0]
    val_ds = ds[1]
    test_ds = ds[2]
    

    print("hello world")