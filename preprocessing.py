import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

# Path to the dataset
dataset_dir = 'archive/dataset'


#
# # Define the preprocessing pipeline
# preprocess_pipeline = transforms.Compose([
#     transforms.RandomApply(torch.nn.ModuleList([
#         transforms.GaussianBlur(kernel_size=13, sigma=(32, 48)),
#     ]), p=0.5),
#     transforms.Resize((224, 224)),  # Resize to 224x224
#     transforms.RandomHorizontalFlip(p=0.2),  # Random horizontal flip
#     transforms.RandomRotation(180),  # Random rotation between -15° and 15°
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
#     transforms.RandomGrayscale(p=0.5),  # Convert to grayscale
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize (for RGB images)
# ])
#
#
# def get_pokemon_data(dataset_dir, batch_size=32):
#     preprocess_pipeline2 = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize the image
#         transforms.RandomHorizontalFlip(),  # Random horizontal flip
#         transforms.RandomRotation(15),  # Random rotation between -15° and 15°
#         transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random brightness and contrast adjustment
#         transforms.RandomGrayscale(p=0.5),  # Convert to grayscale
#         transforms.RandomResizedCrop(180),  # Random crop and resize
#         transforms.ToTensor(),  # Convert image to PyTorch Tensor
#         transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the grayscale image
#     ])
#
#
# # Load the dataset using ImageFolder
# pokemon_dataset = datasets.ImageFolder(root=dataset_dir, transform=preprocess_pipeline)
#
# # Create a DataLoader to iterate through the dataset
# pokemon_loader = DataLoader(pokemon_dataset, batch_size=5, shuffle=True)
#
#
# def show_images(batch):
#     images, labels = batch
#     grid = make_grid(images, nrow=8, padding=2)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(grid.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
#     plt.axis('off')
#     plt.show()
#
#
# # Fetch a batch of images to visualize
# data_iter = iter(pokemon_loader)
# images_batch = next(data_iter)
# #show_images(images_batch)

def load_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')  # Convert to RGBA
    return image


def get_pokemon_data(dataset_dir, batch_size):
    preprocess_pipeline = transforms.Compose([
        # transforms.RandomApply(torch.nn.ModuleList([
        #     transforms.GaussianBlur(kernel_size=13, sigma=(32, 48)),
        # ]), p=0.5),
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomHorizontalFlip(p=0.2),  # Random horizontal flip
        transforms.RandomRotation(20),  # Random rotation between -15° and 15°
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Adjust brightness and contrast
        transforms.RandomGrayscale(p=0.2),  # Convert to grayscale
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize (for RGB images)
    ])

    dataset = datasets.ImageFolder(root=dataset_dir, transform=preprocess_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader, len(dataset.classes)


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
    dataset_dir = 'archive/dataset'  # replace with actual path
    batch_size = 64

    dataloader, classes = get_pokemon_data(dataset_dir, batch_size=64)
    visualize_batch(dataloader, classes)