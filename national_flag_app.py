import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io

import json

# Load the best model path from the JSON file
with open('weights/best_model_info.json', 'r') as f:
    best_model_info = json.load(f)
best_model_path = best_model_info['best_model_path']

# Define the label map
label_map = {0: 'USA', 1: 'Canada', 2: 'France', 3: 'Germany', 4: 'Japan'}

# Load the pre-trained model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(label_map))
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app
st.title('National Flag Identifier')
st.write('Upload an image of a national flag and the app will identify the country.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image = transform(image).unsqueeze(0)
    
    # Predict the country
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        country = label_map[predicted.item()]
    
    st.write(f'The flag belongs to: {country}')
