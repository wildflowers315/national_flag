import unittest
import torch
import torchvision.models as models
import torch.nn as nn

from train_model import dataset

class TestTrainModel(unittest.TestCase):
    def test_dataset(self):
        self.assertTrue(len(dataset) > 0)
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
    
    def test_model(self):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(dataset.label_map))
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(model.fc.out_features, len(dataset.label_map))

if __name__ == '__main__':
    unittest.main()
