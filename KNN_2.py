import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def preprocess_dataset(dataset_dir, batch_size, train_ratio=0.7):
    """
    Prepares train and test DataLoaders with different transformations.
    """
    # Define transformations for train and test datasets
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(root=dataset_dir)
    num_images = len(full_dataset)

    # Split indices into train and test sets
    indices = list(range(num_images))
    np.random.shuffle(indices)
    train_size = int(train_ratio * num_images)
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Apply transformations to subsets
    train_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=dataset_dir, transform=test_transform)

    # Create samplers for DataLoaders
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader, len(full_dataset.classes)


def extract_features(data_loader, feature_extractor, device):
    """
    Extracts features using a specified feature extractor.
    """
    feature_extractor.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = feature_extractor(images)
            features.append(outputs.cpu().numpy())
            labels.extend(targets.numpy())

    return np.vstack(features), np.array(labels)


def train_knn_classifier(train_features, train_labels, n_neighbors=5):
    """
    Trains a KNN classifier using training features and labels.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_features, train_labels)
    return knn


def evaluate_classifier(knn, test_features, test_labels):
    """
    Evaluates a trained classifier and computes accuracy.
    """
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


if __name__ == "__main__":
    # Parameters
    dataset_dir = "archive/dataset"
    batch_size = 32
    train_ratio = 0.7
    n_neighbors = 5

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess dataset and split into train and test sets
    train_loader, test_loader, num_classes = preprocess_dataset(dataset_dir, batch_size, train_ratio)

    print(f"Number of classes: {num_classes}")

    # Load feature extractor (e.g., ResNet18)
    feature_extractor = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    feature_extractor.fc = torch.nn.Identity()  # Remove classification layer
    feature_extractor.to(device)

    # Extract features from train and test sets
    train_features, train_labels = extract_features(train_loader, feature_extractor, device)
    test_features, test_labels = extract_features(test_loader, feature_extractor, device)

    # Train KNN classifier
    knn = train_knn_classifier(train_features, train_labels, n_neighbors)

    # Evaluate classifier on test set
    accuracy = evaluate_classifier(knn, test_features, test_labels)
    print(f"KNN Accuracy: {accuracy:.2f}")
