FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install PyTorch with CUDA support
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]