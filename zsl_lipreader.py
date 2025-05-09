import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob

# Constants
data_dir = 'scripts/data/mouth_rois'
label_set_path = 'scripts/data/label_set.pkl'
label_embeddings_path = 'scripts/data/label_embeddings.pkl'
batch_size = 4
image_size = 96  # ROI size is 64x64

# Dataset
class LipReadingDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.labels = [f.split('_')[1].split('.')[0] for f in self.files]  # filename format: idx_label.npy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.files[idx])  # <- Use self.files here
        label = self.labels[idx]
        video = np.load(video_path)

        # Shape check
        if video.ndim != 3 or video.shape[1:] != (96, 96):
            print(f"[BAD SHAPE] {video_path} has shape {video.shape}, skipping...")
            return self.__getitem__((idx + 1) % len(self.files))  # <- Also use self.files here

        video = torch.tensor(video, dtype=torch.float32).unsqueeze(1) / 255.0  # (T, 1, 96, 96)
        return video, label

def collate_fn(batch):
    videos, labels = zip(*batch)
    max_len = max(v.size(0) for v in videos)
    padded_videos = []
    for v in videos:
        if v.shape[1:] != (1, 96, 96):
            print(f"[BAD SHAPE] {v.shape}")
        pad_len = max_len - v.size(0)
        pad = torch.zeros((pad_len, 1, image_size, image_size))
        padded_videos.append(torch.cat([v, pad], dim=0))
    return torch.stack(padded_videos), labels

# Model
class LipReader(nn.Module):
    def __init__(self, embedding_dim):
        super(LipReader, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(input_size=64 * (image_size // 4) * (image_size // 4), hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.conv(x)
        x = x.view(B, T, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use last time step
        return self.fc(x)

def load_label_embeddings():
    import pickle
    with open(label_embeddings_path, 'rb') as f:
        return pickle.load(f)

def train():
    dataset = LipReadingDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"[INFO] Found {len(dataset)} samples in {data_dir}")

    label_embeddings = load_label_embeddings()
    model = LipReader(embedding_dim=label_embeddings[list(label_embeddings.keys())[0]].shape[0])
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        epoch_loss = 0
        for videos, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(videos)
            target = torch.stack([torch.tensor(label_embeddings[l]) for l in labels])
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss / len(dataloader):.4f}")
    
    # Save the trained model
    model_path = "zsl_lipreader_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}")
    

if __name__ == "__main__":
    train()
