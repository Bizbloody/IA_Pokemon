import torch
from Model import PokemonClassifier, get_resnet_model
from preprocessing import get_pokemon_data
import preprocessing23


def evaluate_model(dataset_dir, model_path, batch_size=32, transfer_learning=False, dataset_separation=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and model
    if dataset_separation:
        pokemon_loader, num_classes = test.test_loader, test.test_size
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


if __name__ == '__main__':
    dataset_dir = 'archive/dataset'
    model_path = 'pokemon_classifier120.0520.pth'
    evaluate_model(dataset_dir, model_path)
