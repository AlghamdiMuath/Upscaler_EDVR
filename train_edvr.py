from matplotlib import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from edvr_model import EDVRModel
from preprocess_video import FrameDataset

# Hyperparameters
BATCH_SIZE = 4  # Lowered for better compatibility on limited resources
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
CHECKPOINT_PATH = "./checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Dataset and Dataloader
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = FrameDataset('./extracted_frames/train', transform=transform)
val_dataset = FrameDataset('./extracted_frames/val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, and Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EDVRModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and Validation Loops
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for low_res, high_res in tqdm(loader, desc="Training"):
        low_res, high_res = low_res.to(device), high_res.to(device)
        optimizer.zero_grad()
        output = model(low_res)
        loss = criterion(output, high_res)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for low_res, high_res in tqdm(loader, desc="Validating"):
            low_res, high_res = low_res.to(device), high_res.to(device)
            output = model(low_res)
            loss = criterion(output, high_res)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# Main Training Loop
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save model checkpoints for recovery
    checkpoint_file = os.path.join(CHECKPOINT_PATH, f"edvr_model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Saved checkpoint: {checkpoint_file}")
