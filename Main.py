from train import train_model
from evaluate import evaluate_model
from preprocessing23 import get_train_test_loaders

dataset_dir = 'archive/dataset'
model_name = 'pokemon_classifier1'
pth = '.pth'
model_directory = f'{model_name + pth}'
train_loader, test_loader, number_classes = get_train_test_loaders(dataset_dir, 64)

train_model(dataset_dir, model_name, train_loader, number_classes)
evaluate_model(dataset_dir, model_directory, test_loader, number_classes)
