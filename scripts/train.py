import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from data.dataset import LaneDataset
from models.model import SimpleCNN

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Data transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Datasets and DataLoaders
train_dataset = LaneDataset(root_dir='images/train', transform=transform)
val_dataset = LaneDataset(root_dir='images/val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss function, optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to visualize predictions
def visualize_predictions(model, dataloader, num_images=4):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, images in enumerate(dataloader):
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            for j in range(images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'Predicted: {preds[j].item()} ({confs[j].item():.2f})')
                plt.imshow(images[j].permute(1, 2, 0).cpu().numpy())

                if images_so_far == num_images:
                    model.train(mode=True)
                    return
                
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        labels = torch.zeros(images.size(0), dtype=torch.long)  # Dummy labels
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images in val_loader:
            outputs = model(images)
            labels = torch.zeros(images.size(0), dtype=torch.long)  # Dummy labels
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss/len(val_loader)}')

    # Visualize predictions after each epoch
    #visualize_predictions(model, val_loader, num_images=4)
   # plt.show()
   # After training loop
torch.save(model.state_dict(), 'lane_detection_model.pth')