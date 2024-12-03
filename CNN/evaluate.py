import torch
from Model import PokemonClassifier, get_resnet_model
from image_show import get_pokemon_data
from preprocessing_CNN import get_train_test_loaders


def evaluate_model(dataset_dir, model_path, train_loader, number_classes, batch_size=32, transfer_learning=True, dataset_separation=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and model
    if dataset_separation:
        pokemon_loader, num_classes = train_loader, number_classes
    else:
        pokemon_loader, num_classes = get_pokemon_data(dataset_dir, batch_size)
    # Initialize the model, loss function, and optimizer
    if transfer_learning:
        model = get_resnet_model(num_classes=num_classes)
    else:
        model = PokemonClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in pokemon_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    return accuracy