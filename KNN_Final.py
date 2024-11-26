import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader


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


def knn_cross_validation(dataset_dir, batch_size=32, k=5, num_folds=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dataset transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    # Initialize a feature extractor (using ResNet18)
    feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    feature_extractor.fc = torch.nn.Identity()  # Remove the classification layer
    feature_extractor.to(device)

    # Initialize cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    print("Performing cross-validation...")
    for fold, (train_indices, test_indices) in enumerate(kf.split(full_dataset)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Create train and test datasets for this fold
        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Extract features and labels for training and testing
        print("  Extracting training features...")
        train_features, train_labels = extract_features(train_loader, feature_extractor, device)
        print("  Extracting testing features...")
        test_features, test_labels = extract_features(test_loader, feature_extractor, device)

        # Train KNN
        print("  Training KNN classifier...")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features, train_labels)

        # Predict on test set
        print("  Predicting test data...")
        predictions = knn.predict(test_features)

        # Evaluate
        accuracy = accuracy_score(test_labels, predictions)
        fold_accuracies.append(accuracy)
        print(f"  Fold {fold + 1} Accuracy: {accuracy:.2f}")

        print("  Classification Report:")
        print(classification_report(test_labels, predictions))

    # Calculate and report overall cross-validation accuracy
    mean_accuracy = np.mean(fold_accuracies)
    print(f"\nOverall Cross-Validation Accuracy: {mean_accuracy:.2f}")
    return mean_accuracy


if __name__ == '__main__':
    dataset_dir = 'archive/dataset'  # Path to your dataset
    mean_accuracy = knn_cross_validation(dataset_dir, batch_size=32, k=5, num_folds=3)
