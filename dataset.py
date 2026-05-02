import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
        root_dir (str): Root directory of the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        num_samples (int, optional): Number of samples to use. If 0, use all samples.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all class names (assuming each subdirectory is a class)
        self.classes = os.listdir(root_dir)
        self.num_classes = len(self.classes)
        
        # Create a dictionary mapping class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Find all image paths
        self.image_paths = self._find_images()

        
        print(f"Loaded {len(self.image_paths)} images from {self.num_classes} classes.")

    def _find_images(self):
        """
        Recursively find all images in the dataset directory.
        
        Returns:
        list: A list of paths to all image files.
        """
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        return [
            os.path.join(root, name)
            for root, _, files in os.walk(self.root_dir)
            for name in files
            if name.lower().endswith(extensions)
        ]

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        
        Args:
        idx (int): Index of the sample to retrieve.
        
        Returns:
        tuple: (image, label) where label is a one-hot encoded tensor.
        """
        # Get the image path and determine its class
        image_path = self.image_paths[idx]
        class_name = os.path.basename(os.path.dirname(image_path))
        class_idx = self.class_to_idx[class_name]
        
        # Open and convert the image to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Apply the transform if it exists
        if self.transform:
            image = self.transform(image)
        
        # Create a one-hot encoded label tensor
        label = torch.zeros(self.num_classes)
        label[class_idx] = 1
        
        return image, label