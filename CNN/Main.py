from train import train_model
from evaluate import evaluate_model
from preprocessing_CNN import get_train_test_loaders


dataset_dir = '../archive/dataset'
model_name = 'pokemon_classifier3.pth'
train_loader, test_loader, number_classes = get_train_test_loaders(dataset_dir, 64)

train_model(dataset_dir, model_name, train_loader, test_loader, number_classes)
evaluate_model(dataset_dir, model_name, test_loader, number_classes)
