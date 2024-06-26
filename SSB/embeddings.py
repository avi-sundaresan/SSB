from torch.utils.data import Dataset
import torch
from tqdm import tqdm 

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        patch_tokens, class_token = self.embeddings[idx]
        label = self.labels[idx]
        return patch_tokens, class_token, label
    
def precompute_embeddings(dataloader, feature_model, device):
    embeddings = []
    labels = []
    feature_model.eval()
    with torch.no_grad():
        for images, lbls in tqdm(dataloader):
            images = images.to(device)
            features = feature_model(images)
            ((patch_tokens, class_token),) = features
            embeddings.append((patch_tokens.clone(), class_token.clone()))
            labels.append(lbls.clone())
    return embeddings, labels
