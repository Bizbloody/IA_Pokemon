import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class DatasetPokemon(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve a single item from your dataset
        item = self.data[index]

        # Load and preprocess the image using PIL and torchvision
        image = Image.open(item['path'])

        if self.transform:
            image = self.transform(image)

            # Get the label
            label = item['label']

            # Return the preprocessed image and label
            return image, label

    # Define the transformation to be applied to the image


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create an instance of your dataset with the transform
data = [{'path': 'archive/dataset/Abra/2eb2a528f9a247358452b3c740df69a0.jpg', 'label': 'abra'}]
dataset = DatasetPokemon(data, transform=transform)

# Create a data loader for your dataset
batch_size = 2
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate over the data loader to get batches of data
# Iterate over the data loader to get batches of data
for images, labels in dataloader:
    # Plot the first image in the batch
    image = images[0].numpy()  # Convert the image tensor to a numpy array
    image = image.transpose(1, 2, 0)  # Transpose the image array to match the expected shape by matplotlib

    # Plot the image using matplotlib
    plt.imshow(image)
    plt.title(f"Label: {labels[0]}")
    plt.axis("on")
    plt.show()
    break  # Break the loop after plotting the first image

# Process the batch
# ...
