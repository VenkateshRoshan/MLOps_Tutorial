import torch
import torch.nn as nn
import torch.optim as optim
from mlflow import log_metric, log_param, log_artifact
from mlflow import start_run
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(train_loader):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with start_run():
        log_param("epochs", 5)
        
        for epoch in range(5):  # Train for 5 epochs
            print(f"Epoch {epoch+1}")
            epoch_loss = []
            for images, labels in train_loader:

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                print(f'\r Epoch : {epoch+1} , Loss: {loss.item()}', end='')
            print(f"\rLoss: {sum(epoch_loss)/len(epoch_loss)}")

            log_metric("loss", loss.item(), step=epoch)
        
        # Save the model
        torch.save(model.state_dict(), 'mnist_model.pth')
        log_artifact('mnist_model.pth')

    return model
