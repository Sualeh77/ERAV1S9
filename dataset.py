import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

torch.manual_seed(1)


class Cifar10Dataset(Dataset):
    """
    Dataset class for Cifar10 dataset
    """

    def __init__(self, dataset, transforms=None):
        """Initialize Dataset

        Args:
            dataset (Dataset): Pytorch Dataset instance
            transforms (Transform.Compose, optional): Tranform function instance. Defaults to None.
        """
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        """
        Get dataset length

        Returns:
            int: Length of dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Get an item form dataset

        Args:
            idx (int): id of item in dataset

        Returns:
            (tensor, int): Return tensor of transformer image, label
        """
        # Read Image and Label
        image, label = self.dataset[index]
        image = np.array(image)

        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        
        return (image, label)
    

def get_loader(train_data, test_data, train_transform, test_transform, batch_size=128, use_cuda=False, use_mps=False):
    """
    Get instance of train and test loaders

    Args:
        train_transform (Transform): Instance of transform function for training
        test_transform (Transform): Instance of transform function for validation
        batch_size (int, optional): batch size to be uised in training. Defaults to 64.
        use_cuda (bool, optional): Enable/Disable Cuda Gpu. Defaults to False.
        use_mps (bool, optional): Enable/Disable MPS for mac. Defaults to False.

    Returns:
        (DataLoader, DataLoader): Get instance of train and test data loaders
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda or use_mps else {}

    train_loader = DataLoader(
            Cifar10Dataset(train_data, transforms=train_transform),
            batch_size=batch_size, shuffle=True, **kwargs
        )
    
    test_loader = DataLoader(
            Cifar10Dataset(test_data, transforms=test_transform),
            batch_size=batch_size, shuffle=True, **kwargs
        )
    
    return train_data, test_data, train_loader, test_loader