import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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


def kmeans_clustering_on_ground_truth(dataset_dir, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dataset transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the entire dataset 
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    # Create DataLoader
    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # Initialize a feature extractor (using ResNet18)
    feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    feature_extractor.fc = torch.nn.Identity()  # Remove the classification layer
    feature_extractor.to(device)

    # Extract features and ground-truth labels
    features, ground_truth_labels = extract_features(data_loader, feature_extractor, device)

    # Perform K-means clustering
    num_clusters = len(set(ground_truth_labels))  # Number of unique classes in the dataset
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Evaluate clustering
    ari = adjusted_rand_score(ground_truth_labels, cluster_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, cluster_labels)

    print(f"Adjusted Rand Index (ARI): {ari:.2f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.2f}")

    return cluster_labels, ground_truth_labels, ari, nmi


if __name__ == '__main__':
    dataset_dir = 'archive/dataset'  # Path to your dataset

    cluster_labels, ground_truth_labels, ari, nmi = kmeans_clustering_on_ground_truth(
        dataset_dir
    )

    print("Cluster Labels:", cluster_labels)
    print("Ground Truth Labels:", ground_truth_labels)