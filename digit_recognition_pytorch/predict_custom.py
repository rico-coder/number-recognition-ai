#
# Author: Rico Krenn
# Created on: Wed Nov 12 2025 11:32:28 AM
# Description: this file predicts the number of images
# This prject is made witch ChatGPT
# File: predict_custom.py
# Workspace: 102_python_projekt
#

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Neural network (must match training network)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model
model = SimpleNN()
model.load_state_dict(torch.load('best_mnist_model.pth'))
model.eval()

def predict_custom_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)

    # Invert if background is white
    if img_array.mean() > 127:
        img_array = 255 - img_array

    img_array = img_array / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    print(f"Prediction: {predicted.item()}, Confidence: {confidence.item()*100:.1f}%")
    return predicted.item()

# Example usage
predict_custom_image('../images_to_test/my_digit_9.png')
