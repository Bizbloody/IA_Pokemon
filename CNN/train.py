import torch
from Model import PokemonClassifier, get_resnet_model
from preprocessing import get_pokemon_data
from preprocessing23 import get_train_test_loaders
import torch.optim as optim
import torch.nn as nn
import time


def train_model(dataset_dir, model_name, train_loader, val_loader, number_classes, num_epochs=100, batch_size=64, lr=0.001, transfer_learning=True, dataset_separation=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data and class count
    if dataset_separation:
        pokemon_loader, num_classes = train_loader, number_classes
    else:
        pokemon_loader, num_classes = get_pokemon_data(dataset_dir, batch_size)

    # Initialize the model, loss function, and optimizer
    if transfer_learning:
        model = get_resnet_model(num_classes=num_classes).to(device)
    else:
        model = PokemonClassifier(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Choose scheduler: StepLR or ReduceLROnPlateau
    use_plateau_scheduler = True  # Set to False if you want to use StepLR
    if use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5,
                                                               verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        # Training phase
        model.train()
        running_loss = 0.0
        batch_start_time = time.time()

        for batch_idx, (images, labels) in enumerate(pokemon_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print batch time every 10 batches
            if (batch_idx + 1) % 10 == 0:
                batch_end_time = time.time()
                elapsed_time = batch_end_time - batch_start_time
                print(f"Batch {batch_idx + 1}: Time for last 10 batches: {elapsed_time:.2f} seconds")
                batch_start_time = time.time()

        avg_train_loss = running_loss / len(pokemon_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation phase (required for ReduceLROnPlateau)
        if use_plateau_scheduler:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:  # Assuming `val_loader` is your validation DataLoader
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
        else:
            # Update learning rate for StepLR
            scheduler.step()

        end_time = time.time()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Duration: {end_time - start_time:.2f} seconds")

    # Save the model
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")


if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    dataset_dir = '../archive/dataset'
    train_model(dataset_dir)
