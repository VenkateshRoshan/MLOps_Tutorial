import pytest
import json
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


# from app import app

# def test_predict():
#     client = app.test_client()
#     response = client.post('/predict', json={'input': [[0.0] * 28] * 28})
#     assert response.status_code == 200
#     assert 'prediction' in json.loads(response.data)
