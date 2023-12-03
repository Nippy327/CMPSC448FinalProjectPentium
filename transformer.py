import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch.optim as optim
import torch.nn as nn
import evaluate

# Set up data loaders
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

train_dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=3, ignore_mismatched_sizes=True)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    print(epoch)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation on validation set (similar to training loop but without backpropagation)
# Evaluate the model on data/test set and store the predicted labels in a dictionary
val_dataset = datasets.ImageFolder('data/test', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
pred_labels = {}
model.eval()
iterp = 1
for inputs, labels in val_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs).logits
    _, predicted = torch.max(outputs, 1)
    for i in range(len(predicted)):
        pred_labels[i] = predicted[i].item()
        if pred_labels[i] == 0:
            labeltxt = "cloudy"
        elif pred_labels[i] == 1:
            labeltxt = "rain"
        else:
            labeltxt = "shine"
        print(f"Label {iterp} is {labeltxt}")
        pred_labels[iterp] = labeltxt
        iterp += 1

# Evaluate the predicted labels
accuracy = evaluate.evaluate(pred_labels)
print("Accuracy: ", accuracy)