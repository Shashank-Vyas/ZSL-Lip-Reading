import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle

# Constants
test_data_dir = 'scripts/data/mouth_rois/test_rois'  # Change if your test data is elsewhere
label_embeddings_path = 'scripts/data/label_embeddings.pkl'
model_path = 'zsl_lipreader_model_new.pth'
image_size = 96
batch_size = 4

# Dataset
class LipReadingDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.labels = [f.split('_')[1].split('.')[0] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.files[idx])
        label = self.labels[idx]
        video = np.load(video_path)

        if video.ndim != 3 or video.shape[1:] != (96, 96):
            return self.__getitem__((idx + 1) % len(self.files))

        video = torch.tensor(video, dtype=torch.float32).unsqueeze(1) / 255.0  # (T, 1, 96, 96)
        return video, label

def collate_fn(batch):
    videos, labels = zip(*batch)
    max_len = max(v.size(0) for v in videos)
    padded_videos = []
    for v in videos:
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
        x = x[:, -1, :]
        return self.fc(x)

def load_label_embeddings():
    with open(label_embeddings_path, 'rb') as f:
        return pickle.load(f)

def evaluate():
    dataset = LipReadingDataset(test_data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"[INFO] Found {len(dataset)} test samples in {test_data_dir}")

    label_embeddings = load_label_embeddings()
    label_names = list(label_embeddings.keys())
    label_matrix = np.stack([label_embeddings[l] for l in label_names])  # (num_labels, emb_dim)

    model = LipReader(embedding_dim=label_matrix.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Evaluating"):
            outputs = model(videos)  # (B, embedding_dim)
            outputs_np = outputs.cpu().numpy()
            sims = cosine_similarity(outputs_np, label_matrix)  # (B, num_labels)
            preds = [label_names[i] for i in np.argmax(sims, axis=1)]

            for pred, actual in zip(preds, labels):
                if pred == actual:
                    correct += 1
                total += 1

    accuracy = correct / total * 100
    print(f"[RESULT] Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()
