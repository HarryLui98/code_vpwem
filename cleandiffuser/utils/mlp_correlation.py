import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def batch_mlp_corr(actions, num_epochs=20, batch_size=2048, val_ratio=0.2):
    # actions with shape (num_envs, sequence, action_dim)
    actions = torch.from_numpy(actions).float().cuda()
    N, S, dim = actions.shape
    model = MLP(dim, 512, dim).to(actions.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = actions[:, :-1, :]  # Shape: (N, S-1, dim)
    y = actions[:, 1:, :]   # Shape: (N, S-1, dim)

    # Reshape to combine batch and sequence dimensions
    x = x.reshape(-1, dim)  # Shape: ((N*(S-1)), dim)
    y = y.reshape(-1, dim)  # Shape: ((N*(S-1)), dim)

    # Split x, y into train/val datasets
    dataset = TensorDataset(x, y)
    num_val = int(len(dataset) * val_ratio)
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Track losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(actions.device), yb.to(actions.device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float("inf")
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(actions.device), yb.to(actions.device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # # Plot training and validation loss
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    # plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="s")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Training and Validation Loss Curve")
    # plt.legend()
    # plt.grid()
    # plt.savefig("loss.png")
    # plt.show()

    # Return the average of the 5 lowest validation scores
    return sorted(val_losses)[0]

def batch_mlp_corr_dataset(actions, num_epochs=20, batch_size=2048, val_ratio=0.2):
    # actions with shape: list of (sequence, action_dim)
    # actions = torch.from_numpy(actions).float().cuda()
    # N, S, dim = actions.shape
    device = 'cpu'
    dim = actions[0].shape[1]
    model = MLP(dim, 512, dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    all_x = []
    all_y = []
    for action in actions:
        x = action[:-1, :]  # Shape: (N, S-1, dim)
        y = action[1:, :]   # Shape: (N, S-1, dim)
        all_x.append(x)
        all_y.append(y)

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    # Reshape to combine batch and sequence dimensions
    # x = x.reshape(-1, dim)  # Shape: ((N*(S-1)), dim)
    # y = y.reshape(-1, dim)  # Shape: ((N*(S-1)), dim)
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    # Split x, y into train/val datasets
    dataset = TensorDataset(x, y)
    num_val = int(len(dataset) * val_ratio)
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Track losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float("inf")
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float("inf")
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # # Plot training and validation loss
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    # plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="s")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Training and Validation Loss Curve")
    # plt.legend()
    # plt.grid()
    # plt.savefig("loss.png")
    # plt.show()

    # Return the average of the 5 lowest validation scores
    return sorted(val_losses)[0], model