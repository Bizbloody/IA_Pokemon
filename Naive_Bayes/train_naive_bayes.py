from PCA import pca_workflow
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define dataset directory
dataset_dir = "../archive/dataset"

# Print start of PCA workflow
print("Starting PCA workflow...")

# Perform PCA workflow
X_train_pca, X_test_pca, y_train, y_test, pca = pca_workflow(
    dataset_dir=dataset_dir,
    train_ratio=0.7,
    batch_size=64,
    n_components=50,
    save_plot=True
)

# Print PCA details
print("PCA workflow completed!")
print(f"Number of components used: {pca.n_components_}")
print("Explained variance (cumulative):")
print(pca.explained_variance_ratio_.cumsum())

# Compute class weights
print("\nComputing class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Create a sample weight array for training data
sample_weights = np.array([class_weights[class_idx] for class_idx in y_train])

# Print start of model training
print("\nStarting Naive Bayes model training...")

# Train Naive Bayes model with sample weights
clf = GaussianNB()
clf.fit(X_train_pca, y_train, sample_weight=sample_weights)

# Print end of training
print("Model training completed!")

# Print start of predictions
print("\nMaking predictions on the test set...")
y_pred = clf.predict(X_test_pca)

# Print evaluation metrics
print("\nEvaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
