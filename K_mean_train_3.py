import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
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


def apply_pca(features, n_components=50):
    """
    Reduces dimensionality of features using PCA.
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA Explained Variance: {explained_variance:.2f}%")
    return reduced_features


def calculate_accuracy(cluster_labels, ground_truth_labels):
    """
    Calculates accuracy , map cluster labels to ground truth labels.
    """
    # Create a confusion matrix
    num_classes = len(set(ground_truth_labels))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    for gt_label, cluster_label in zip(ground_truth_labels, cluster_labels):
        confusion_matrix[gt_label, cluster_label] += 1

    # Find the best mapping
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)  # Maximize matching

    # Compute accuracy
    correct_predictions = confusion_matrix[row_ind, col_ind].sum()
    total_samples = len(ground_truth_labels)
    accuracy = correct_predictions / total_samples

    return accuracy


def kmeans_clustering_on_ground_truth(dataset_dir, batch_size=32, n_components=50):
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

    # Apply PCA
    reduced_features = apply_pca(features, n_components)

    # Perform K-means clustering
    num_clusters = len(set(ground_truth_labels))  # Number of unique classes in the dataset
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # Evaluate clustering
    ari = adjusted_rand_score(ground_truth_labels, cluster_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, cluster_labels)
    accuracy = calculate_accuracy(cluster_labels, ground_truth_labels)

    print(f"Adjusted Rand Index (ARI): {ari:.2f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    return cluster_labels, ground_truth_labels, ari, nmi, accuracy


if __name__ == '__main__':
    dataset_dir = 'archive/dataset'  # Path to your dataset

    cluster_labels, ground_truth_labels, ari, nmi, accuracy = kmeans_clustering_on_ground_truth(
        dataset_dir, n_components=100
    )

    print("Cluster Labels:", cluster_labels)
    print("Ground Truth Labels:", ground_truth_labels)
