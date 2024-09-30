
from flask import Flask, request, jsonify, render_template
import torch
from model import SimpleCNN
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Define the transformation to apply to the uploaded image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model loading
model = SimpleCNN()
model.load_state_dict(torch.load('mnist_model.pth')).to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = transform(img)
        input_tensor = img.unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1, keepdim=True).item()
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)