# Step 1: Define the Task
**Objective**: Create a Streamlit app that allows users to upload an image of a national flag in United nation and identifies the country using a deep learning model.  
**Input**: An image file (e.g., JPEG, PNG) uploaded by the user.  
**Output**: The name of the country whose flag is in the image.

# Step 2: Set Up the Environment
- use conda to create national_flag environment with python =>3.10
- Install Streamlit: `pip install streamlit`
- Install PyTorch: `pip install torch torchvision`
- Install OpenCV: `pip install opencv-python`
- Install Pillow: `pip install pillow`

# Step 3: Prepare the Dataset
- **Collect Flag Images**: Gather a dataset of national flag images.
- **Label the Data**: Ensure each image is labeled with the corresponding country name.
- **Preprocess the Data**: Resize images, normalize pixel values, and split into training and testing sets.

## Data Collection Algorithm
To collect images from the internet, you can use the Google Custom Search API. Below is an example of how to use it to collect flag images:

1. **Set Up Google Custom Search API**:
   - You can see in `api_key` file.

2. **Install Required Libraries**:
   ```sh
   pip install google-api-python-client
   ```

```python
from googleapiclient.discovery import build
import requests
import os

# Set up the API key and custom search engine ID
api_key = 'YOUR_API_KEY'
cse_id = 'YOUR_CSE_ID'

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
search_term = 'national flag of USA'
items = google_search(search_term, api_key, cse_id)
download_images(items, 'flags/USA')
```

3. **Collect 10 images per one national flag**
Use sources like Wikipedia, live photos, or more real-world photos to ensure diversity in the dataset.
If numbers of request is beyond the limit of api, stop there.

# Step 4: Build the Deep Learning Model
Choose a Model Architecture: Use a pre-trained model like VGG16, ResNet, or a custom CNN.
Train the Model: Train the model on the flag dataset.
Save the Model: Save the trained model to a file for later use.

# Step 5: Create the Streamlit App
Set Up the App Structure: Create a new Python file for the Streamlit app.
Upload Image: Add a file uploader widget to allow users to upload images.
Load the Model: Load the pre-trained model.
Predict the Flag: Use the model to predict the country from the uploaded image.
Display the Result: Show the prediction result to the user.

# Step 6: Test
Test the App: Run the app locally and test with various flag images.