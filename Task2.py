# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

# ==============================
# 2. DEVICE SETUP (GPU/CPU)
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 3. DATASET PATH (CHANGE HERE)
# ==============================
# 👉 CHANGE THIS PATH to your dataset folder
DATASET_PATH = "C:/Users/hrida/Downloads/archive (3)/original"   # e.g. "C:/Users/YourName/waste_dataset"

# ==============================
# 4. DATA PREPROCESSING
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize images
    transforms.RandomHorizontalFlip(),      # Data augmentation
    transforms.RandomRotation(10),          # Data augmentation
    transforms.ToTensor(),                  # Convert to tensor
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize
])

# Load dataset
train_data = datasets.ImageFolder(root=f"{DATASET_PATH}/train", transform=transform)
test_data = datasets.ImageFolder(root=f"{DATASET_PATH}/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Class names (automatic)
class_names = train_data.classes
print("Classes:", class_names)

# ==============================
# 5. LOAD PRETRAINED MODEL
# ==============================
model = models.resnet18(pretrained=True)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = len(class_names)   # 👉 Automatically detects 3 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# ==============================
# 6. LOSS & OPTIMIZER
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ==============================
# 7. TRAINING LOOP
# ==============================
EPOCHS = 10   # 👉 CHANGE if needed

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    loss_history.append(running_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

# ==============================
# 8. MODEL EVALUATION
# ==============================
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# ==============================
# 9. CONFUSION MATRIX ✅ (FIXED)
# ==============================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()

# ==============================
# 10. LOSS GRAPH
# ==============================
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
# ==============================
# 11. SAVE MODEL
# ==============================
torch.save(model.state_dict(), "waste_classifier.pth")
print("Model saved!")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


