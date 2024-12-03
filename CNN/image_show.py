import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

# Path to the dataset
dataset_dir = '../archive/dataset'

def load_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')  # Convert to RGBA
    return image


def get_pokemon_data(dataset_dir, batch_size):
    preprocess_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomApply(torch.nn.ModuleList([
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

    dataset = datasets.ImageFolder(root=dataset_dir, transform=preprocess_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader, dataset.classes


def visualize_batch(data_loader, classes, batch_size=8):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)  # Retrieve one batch of data

    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))
    for i in range(batch_size):
        image = images[i].permute(1, 2, 0).numpy()  # Convert to numpy for display
        label = classes[labels[i].item()]
        axes[i].imshow(image)
        axes[i].set_title(label)
        axes[i].axis('off')

    plt.show()
    print("Batch labels:", [classes[label.item()] for label in labels[:batch_size]])


if __name__ == '__main__':
    # Your main code execution
    dataset_dir = '../archive/dataset'  # replace with actual path
    batch_size = 64

    dataloader, classes = get_pokemon_data(dataset_dir, batch_size=64)
    visualize_batch(dataloader, classes)