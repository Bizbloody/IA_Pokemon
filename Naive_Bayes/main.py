from load_data import load_and_preprocess_data
from dimensionality_reduction import apply_pca
from train_naive_bayes import train_and_evaluate_gnb

# Example placeholders for image and label datasets
# Replace these with actual dataset loading logic
images = ...  # Load your image dataset as a NumPy array
labels = ...  # Load your corresponding labels as a NumPy array

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(images, labels)

    # Apply dimensionality reduction
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    # Train and evaluate Gaussian Naive Bayes
    accuracy = train_and_evaluate_gnb(X_train_pca, X_test_pca, y_train, y_test)
    print(f"Accuracy of Gaussian Naive Bayes: {accuracy:.2f}")

if __name__ == "__main__":
    main()
