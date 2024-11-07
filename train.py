import torch

import preprocessing23
from Model import PokemonClassifier, get_resnet_model
from preprocessing import get_pokemon_data
from preprocessing23 import train_dataset, train_loader
import torch.optim as optim
import torch.nn as nn
import time


def train_model(dataset_dir, num_epochs=100, batch_size=64, lr=0.001, transfer_learning=False, dataset_separation=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data and class count
    if dataset_separation:
        pokemon_loader, num_classes = train_loader, test.train_size
    else:
        pokemon_loader, num_classes = get_pokemon_data(dataset_dir, batch_size)

    # Initialize the model, loss function, and optimizer
    if transfer_learning:
        model = get_resnet_model(num_classes=num_classes).to(device)
    else:
        model = PokemonClassifier(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Decay LR every 10 epochs
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # Alternatively, based on loss plateau

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        model.train()
        running_loss = 0.0

        batch_start_time = time.time()

        # Iterate over batches
        for batch_idx, (images, labels) in enumerate(pokemon_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Check if 10 batches have completed
            if (batch_idx + 1) % 10 == 0:
                batch_end_time = time.time()
                elapsed_time = batch_end_time - batch_start_time
                print(f"Batch {batch_idx + 1}: Time for last 10 batches: {elapsed_time:.2f} seconds")

                # Reset the timer for the next 10 batches
                batch_start_time = time.time()

        scheduler.step(running_loss / len(pokemon_loader) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None) # Update learning rate

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(pokemon_loader):.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Duration: {end_time - start_time:.2f} seconds')

    # Save the model
    torch.save(model.state_dict(), 'pokemon_classifier120.0520.pth')
    print("Model saved as pokemon_classifier120.0520.pth")


if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    dataset_dir = 'archive/dataset'
    train_model(dataset_dir)
