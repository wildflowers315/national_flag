import unittest
import os
import torch
from collect_and_preprocess_data import google_search, download_images, FlagDataset

class TestCollectAndPreprocessData(unittest.TestCase):
    def test_google_search(self):
        api_key = 'AIzaSyBd3ah8fkELXKb6N3WCPV_qEvsuP8FW4Tg'
        cse_id = 'my-project-423921'
        search_term = 'national flag of USA'
        items = google_search(search_term, api_key, cse_id, num_images=1)
        self.assertTrue(len(items) > 0)
    
    def test_download_images(self):
        items = [{'link': 'https://upload.wikimedia.org/wikipedia/en/a/a4/Flag_of_the_United_States.svg'}]
        download_images(items, 'test_flags/USA')
        self.assertTrue(os.path.exists('test_flags/USA/img_0.jpg'))
    
    def test_flag_dataset(self):
        transform = None
        dataset = FlagDataset(root_dir='flags', transform=transform)
        self.assertTrue(len(dataset) > 0)
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)

if __name__ == '__main__':
    unittest.main()
