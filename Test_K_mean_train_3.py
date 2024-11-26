import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def filter_dataset(dataset, target_classes):
    """
    Filters a dataset to include only the specified target classes.
    """
    filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in target_classes]
    return Subset(dataset, filtered_indices)


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

    features = np.vstack(features)
    return features, ground_truth_labels

def apply_pca(features, n_components=50):
    """
    Apply PCA to reduce the dimensionality of the feature set.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)  # Apply PCA and reduce dimensions
    return reduced_features

def kmeans_clustering_on_ground_truth(dataset_dir, target_classes, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define dataset transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load and filter dataset
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    filtered_dataset = filter_dataset(full_dataset, target_classes)

    # Create DataLoader
    data_loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

    # Initialize a feature extractor (using ResNet18)
    feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    feature_extractor.fc = torch.nn.Identity()  # Remove the classification layer
    feature_extractor.to(device)

    # Extract features and ground-truth labels
    features, ground_truth_labels = extract_features(data_loader, feature_extractor, device)
    
    # Apply PCA to reduce dimensionality
    reduced_features = apply_pca(features, n_components=10)

    # Perform K-means clustering
    num_clusters = len(target_classes)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Evaluate clustering
    ari = adjusted_rand_score(ground_truth_labels, cluster_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, cluster_labels)

    print(f"Adjusted Rand Index (ARI): {ari:.2f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.2f}")

    return cluster_labels, ground_truth_labels, ari, nmi


if __name__ == '__main__':
    dataset_dir = 'archive/dataset'
    target_classes = [0, 1, 2, 3, 4]  # Specify 5 Pokémon class indices

    cluster_labels, ground_truth_labels, ari, nmi = kmeans_clustering_on_ground_truth(
        dataset_dir, target_classes=target_classes
    )

    print("Cluster Labels:", cluster_labels)
    print("Ground Truth Labels:", ground_truth_labels)
