import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def extract_features(dataset_loader, feature_extractor, device):
    """
    Extracts features from a dataset using the specified feature extractor model.
    """
    feature_extractor.eval()
    features = []
    ground_truth_labels = []

    with torch.no_grad():
        for images, labels in dataset_loader:
            images = images.to(device)
            outputs = feature_extractor(images)
            features.append(outputs.cpu().numpy())
            ground_truth_labels.extend(labels.numpy())

    features = np.vstack(features)  # Stack features into a single numpy array
    return features, ground_truth_labels


def knn_classification(dataset_dir, train_ratio=0.8, batch_size=32, k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dataset transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    
    # Split dataset into train and test sets
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize a feature extractor (using ResNet18)
    feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    feature_extractor.fc = torch.nn.Identity()  # Remove the classification layer
    feature_extractor.to(device)

    # Extract features and labels for training and testing
    print("Extracting training features...")
    train_features, train_labels = extract_features(train_loader, feature_extractor, device)
    print("Extracting testing features...")
    test_features, test_labels = extract_features(test_loader, feature_extractor, device)

    # Train KNN
    print("Training KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)

    # Predict on test set
    print("Predicting test data...")
    predictions = knn.predict(test_features)

    # Evaluate
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(test_labels, predictions))

    return knn, train_features, train_labels, test_features, test_labels, predictions


if __name__ == '__main__':
    dataset_dir = 'archive/dataset'  # Path to your dataset
    knn, train_features, train_labels, test_features, test_labels, predictions = knn_classification(
        dataset_dir, train_ratio=0.8, batch_size=32, k=5
    )
