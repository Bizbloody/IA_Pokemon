from torchvision import models
from sklearn.cluster import KMeans
import torch
import numpy as np

def extract_features(data_loader, feature_extractor, device):
    """
    Extracts features from a dataset using a pretrained model.
    """
    feature_extractor.eval()
    features = []
    labels = []  # Optional: store labels if available
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = feature_extractor(inputs)
            features.append(outputs.cpu().numpy())
            labels.extend(targets.numpy())
    
    features = np.vstack(features)  # Combine all features into a single array
    return features, labels

def perform_kmeans(features, num_clusters):
    """
    Applies K-means clustering to the features.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels

# Main workflow
def main_kmeans_clustering(dataset_dir, batch_size, num_clusters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_loader, test_loader, num_classes = get_train_test_loaders(dataset_dir, batch_size)

    # Use a pretrained model for feature extraction (e.g., ResNet18)
    pretrained_model = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1])  # Remove the classifier
    feature_extractor = feature_extractor.to(device)

    # Extract features
    train_features, _ = extract_features(train_loader, feature_extractor, device)

    # Flatten features
    train_features = train_features.reshape(train_features.shape[0], -1)

    # Apply K-means
    kmeans, cluster_labels = perform_kmeans(train_features, num_clusters)

    return kmeans, cluster_labels

# Example usage
dataset_dir = 'archive/dataset'
batch_size = 32
num_clusters = 5

kmeans, cluster_labels = main_kmeans_clustering(dataset_dir, batch_size, num_clusters)

# Optional: Print cluster centers or cluster labels
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Cluster labels:\n", cluster_labels)
