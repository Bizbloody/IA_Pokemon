from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from torch import nn
import numpy as np

dataset_dir = '../archive/dataset'


def get_train_test_loaders(dataset_dir, batch_size, train_ratio=0.7):
    # Define data transformations
    preprocess_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomApply(nn.ModuleList([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ]), p=0.5),
        transforms.RandomRotation(20),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load full dataset
    dataset = datasets.ImageFolder(root=dataset_dir, transform=preprocess_pipeline)
    num_images = len(dataset)

    # Prepare indices and shuffle them
    indices = list(range(num_images))
    np.random.shuffle(indices)

    # Compute split indices
    train_size = int(train_ratio * num_images)
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader, len(dataset.classes)
