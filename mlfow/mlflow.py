import mlflow
from ..model import SimpleCNN
import torch

# set uri
mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = SimpleCNN()
# model loading
model = SimpleCNN()
model.load_state_dict(torch.load('../mnist_model.pth'))
model.eval()

# Start a new MLflow run
with mlflow.start_run():
    # Log parameters, metrics, and artifacts
    mlflow.log_param("model_type", "CNN")
    mlflow.log_param("epochs", 5)
    
    # After training the model, log the trained model
    mlflow.pytorch.log_model(model, "mnist_model")