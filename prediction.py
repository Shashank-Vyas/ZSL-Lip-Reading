import torch
import torch.nn.functional as F
import pickle
from zsl_lipreader import LipReadingDataset, collate_fn  # reuse your Dataset
from zsl_lipreader import LipReader  # reuse your model class
from tqdm import tqdm

# Load model
model = LipReader(embedding_dim=300)  # or whatever you used
model.load_state_dict(torch.load("zsl_lipreader_model.pth"))  # Load your trained model
model.eval()

# Load test data
test_dataset = LipReadingDataset("scripts/data/mouth_rois/test_rois")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

# Load test label embeddings
with open("scripts/data/label_embeddings.pkl", "rb") as f:
    label_embeddings = pickle.load(f)
label_names = list(label_embeddings.keys())
label_vectors = torch.stack([torch.tensor(label_embeddings[l]) for l in label_names])  # shape: (N_labels, dim)

# Predict
correct = 0
total = 0

for videos, true_labels in tqdm(test_loader):
    with torch.no_grad():
        video_embedding = model(videos)  # shape: (1, dim)
        similarities = F.cosine_similarity(video_embedding, label_vectors)
        predicted_index = similarities.argmax().item()
        predicted_label = label_names[predicted_index]
        if predicted_label == true_labels[0]:
            correct += 1
        total += 1

print(f"[Zero-Shot Accuracy] {correct}/{total} = {correct / total:.2%}")
