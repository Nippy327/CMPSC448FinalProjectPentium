import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Assuming you have a function to load and preprocess your dataset
def load_dataset(data_dir):
    # Your dataset loading and preprocessing logic here
    # ...

# Custom Transformer Model
class CustomTransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CustomTransformerModel, self).__init__()
        # Define your transformer layers and other components
        # For simplicity, you can use basic linear layers and positional encoding
        self.embedding = nn.Linear(input_size, 512)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Data Loading and Preprocessing
data_dir = '/path/to/your/data'
batch_size = 32

# Assuming you have organized your data into train and test folders
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=train_transform)
test_dataset = datasets.ImageFolder(root=data_dir + '/test', transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Assuming you have three classes: pizza, sandwich, ice cream
num_classes = 3

# Initialize your model, loss function, and optimizer
model = CustomTransformerModel(input_size=224*224*3, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.view(batch_size, -1).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation on the test set
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.view(batch_size, -1).to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Evaluation on validation set (similar to training loop but without backpropagation)
model.eval()
pred_labels = {}
for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs).logits
    _, preds = torch.max(outputs, dim=1)
    for i in range(len(preds)):
        pred_labels[i] = preds[i].item()

# Calculate Accuracy
accuracy = evaluate.evaluate(pred_labels)
print(f"Accuracy: {accuracy}")
