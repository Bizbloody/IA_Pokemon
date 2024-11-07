from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets


dataset_dir = 'archive/dataset'


# Define data transformations for training and testing
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset and apply transformations
dataset = datasets.ImageFolder(root=dataset_dir)

# Calculate sizes for train and test split
train_size = int(0.67 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into train and test subsets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply respective transformations
train_dataset.dataset.transform = train_transforms
test_dataset.dataset.transform = test_transforms

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
