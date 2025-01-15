from googleapiclient.discovery import build
import requests
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from credentials import YOUR_API_KEY, YOUR_CSE_ID
# Set up the API key and custom search engine ID
api_key = YOUR_API_KEY
cse_id = YOUR_CSE_ID

def google_search(search_term, api_key, cse_id, num_images=10):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, searchType='image', num=num_images).execute()
    return res['items']

def download_images(items, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, item in enumerate(items):
        img_url = item['link']
        img_data = requests.get(img_url).content
        with open(os.path.join(folder_path, f'img_{i}.jpg'), 'wb') as handler:
            handler.write(img_data)

# Example usage
countries = ['USA', 'Canada', 'France', 'Germany', 'Japan']
for country in countries:
    search_term = f'national flag of {country}'
    items = google_search(search_term, api_key, cse_id)
    download_images(items, f'flags/{country}')

# Preprocess the data
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
                self.image_paths.append(os.path.join(country_dir, img_name))
                self.labels.append(self.label_map[country])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = FlagDataset(root_dir='flags', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Save the dataset
torch.save(dataset, 'weights/flag_dataset_{}.pth'.format(len(dataset)))
