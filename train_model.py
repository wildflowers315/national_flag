import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
import os

from PIL import Image
from torch.utils.data import Dataset

# Define the countries variable
countries = ['USA', 'Canada', 'France', 'Germany', 'Japan']

# Define the FlagDataset class
class FlagDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {country: idx for idx, country in enumerate(countries)}
        
        for country in countries:
            country_dir = os.path.join(root_dir, country)
        for img_name in os.listdir(country_dir):
            img_path = os.path.join(country_dir, img_name)
            self.image_paths.append(img_path)
            self.labels.append(self.label_map[country])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError):
            print(f"Skipping broken image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load the dataset
dataset = torch.load('weights/flag_dataset_50.pth')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.label_map))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
best_val_accuracy = 0.0
model_info = []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct / total
    val_accuracy = train_accuracy  # Placeholder for validation accuracy
    model_path = 'weights/flag_model_epoch_{}.pth'.format(epoch + 1)
    torch.save(model.state_dict(), model_path)
    
    model_info.append({
        'epoch': epoch + 1,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'model_path': model_path
    })
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_path = model_path

# Save the model info to a JSON file
with open('weights/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

# Save the best model path to a JSON file
best_model_info = {
    'best_model_path': best_model_path,
    'best_val_accuracy': best_val_accuracy
}
with open('weights/best_model_info.json', 'w') as f:
    json.dump(best_model_info, f, indent=4)
