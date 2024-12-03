import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets

class PokemonClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PokemonClassifier, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Adding a fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Pooling and Dropout layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        self.fc1 = nn.Linear(512, num_classes)  # Adjusted for Global Average Pooling

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.pool(self.bn5(F.relu(self.conv5(x))))

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 512)  # Flatten the tensor for FC layer

        x = self.dropout(x)
        x = self.fc1(x)
        return x


# Load the saved model
num_classes = 149  # Adjust to match the number of classes you trained the model on
model = PokemonClassifier(num_classes=num_classes)
model.load_state_dict(torch.load('**model_name**'))
model.eval()  # Set model to evaluation mode

# Define the same transformation pipeline used in training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_dir = 'archive/dataset'

# Load and preprocess the test image
image_path = '$_57.JPG'  # Replace with your image path
image = Image.open(image_path).convert('RGB')  # Ensure 3 channels (RGB)
image = preprocess(image)
image = image.unsqueeze(0)  # Add batch dimension

# Run prediction
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_index = predicted.item()


dataset = datasets.ImageFolder(root=dataset_dir)

# Load class names if you have them saved, or define them directly here
classes = dataset.classes  # Replace with your list of 151 Pokémon class names in alphabetical order
# Print the predicted class index and the length of the classes list for debugging
print(f'Predicted class index: {class_index}')
print(f'Number of classes: {len(classes)}')

# Ensure the index is within range
if class_index < len(classes):
    print(f'Predicted Pokémon: {classes[class_index]}')
else:
    print(f'Error: Predicted class index {class_index} is out of range.')

# Print the predicted class name
print(f'Predicted Pokémon: {classes[class_index]}')
