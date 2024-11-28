from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import SubsetRandomSampler



def get_preprocessed_dataset(dataset_dir):
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Load dataset with preprocessing pipeline
    dataset = datasets.ImageFolder(root=dataset_dir, transform=preprocess_pipeline)
    return dataset


def dataloader_to_numpy(dataloader):
    features = []
    labels = []

    for images, targets in dataloader:
        # Flatten the images and convert to NumPy
        features.append(images.view(images.size(0), -1).numpy())
        labels.append(targets.numpy())

    # Concatenate all batches into a single array
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels




def get_train_test_data(dataset, train_ratio=0.7, batch_size=64):
    num_images = len(dataset)
    indices = list(range(num_images))
    np.random.shuffle(indices)

    # Split indices
    train_size = int(train_ratio * num_images)
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Create loaders
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Convert to NumPy arrays
    X_train, y_train = dataloader_to_numpy(train_loader)
    X_test, y_test = dataloader_to_numpy(test_loader)

    return X_train, X_test, y_train, y_test
