from data_loader import get_data_loaders
from model import train_model

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    # print train and test loaders
    print(train_loader)
    print(test_loader)
    print("Training the model...")
    model = train_model(train_loader)
