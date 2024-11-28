from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from preprocessing_NB import get_preprocessed_dataset, get_train_test_data  # Import your preprocessing logic


def apply_pca(X_train, X_test, n_components=50):

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def evaluate_pca_effectiveness(pca, save_plot=False):

    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(explained_variance_ratio, marker='o', linestyle='--')
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()

    if save_plot:
        plt.savefig("pca_explained_variance.png")
    else:
        plt.show()


def pca_workflow(dataset_dir, train_ratio=0.7, batch_size=64, n_components=50, save_plot=False):

    # Step 1: Preprocess the dataset
    dataset = get_preprocessed_dataset(dataset_dir)
    X_train, X_test, y_train, y_test = get_train_test_data(dataset, train_ratio, batch_size)

    # Step 2: Apply PCA
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components)

    # Step 3: Evaluate PCA effectiveness
    evaluate_pca_effectiveness(pca, save_plot)

    return X_train_pca, X_test_pca, y_train, y_test, pca
