import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm
from functools import partial

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score

from SSB.get_osr_splits.manage_cub import CUBCustomDataset, get_cub_splits
from SSB.models import ModelWithIntermediateLayers, AttentiveClassifier
from SSB.embeddings import EmbeddingsDataset, precompute_embeddings

def train(lr, accum_iter, num_epochs):
    model = AttentiveClassifier(embed_dim=1536, num_classes=10000).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (patch_tokens, class_token, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
            patch_tokens = patch_tokens.to(device)
            class_token = class_token.to(device)
            labels = labels.to(device)

            outputs = model((patch_tokens, class_token))
            loss = criterion(outputs, labels)
            loss.backward()

            train_loss += loss.item()

            if (batch_idx + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
    
        print(f'Epoch {e+1}/{num_epochs}')
        print(f'Training Loss: {train_loss / len(train_labels):.4f}\n')
    return model

train_list, test_list, easy_oos_list, med_oos_list, hard_oos_list, label_map = get_cub_splits()

transformation = transforms.Compose([
        transforms.Resize((224, 224), interpolation= PIL.Image.Resampling.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet default mean and sd
    ])

train_dataset = CUBCustomDataset(train_list, transformation)
test_dataset = CUBCustomDataset(test_list, transformation)
easy_otest_dataset = CUBCustomDataset(easy_oos_list, transformation)
med_otest_dataset = CUBCustomDataset(med_oos_list, transformation)
hard_otest_dataset = CUBCustomDataset(hard_oos_list, transformation)

batch_size = 8

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
easy_opentestloader = torch.utils.data.DataLoader(easy_otest_dataset, batch_size=batch_size, shuffle=True)
med_opentestloader = torch.utils.data.DataLoader(med_otest_dataset, batch_size=batch_size, shuffle=True)
hard_opentestloader = torch.utils.data.DataLoader(hard_otest_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)
n_last_blocks = 1
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
feature_model = ModelWithIntermediateLayers(dinov2_vitg14, n_last_blocks, autocast_ctx).to("cuda")

train_embeddings, train_labels = precompute_embeddings(trainloader, feature_model)
test_embeddings, test_labels = precompute_embeddings(testloader, feature_model)

train_embeddings_dataset = EmbeddingsDataset(train_embeddings, train_labels)
test_embeddings_dataset = EmbeddingsDataset(test_embeddings, test_labels)

trainloader = DataLoader(train_embeddings_dataset, batch_size=None, shuffle=True)
testloader = DataLoader(test_embeddings_dataset, batch_size=None, shuffle=True)

lr = 1e-5
accum_iter = 1
num_epochs = 10
model = train(lr, accum_iter, num_epochs)
model.eval()

total_samples = 0
correct_predictions = 0

all_preds = []
all_labels = []

with torch.no_grad():  
    for batch_idx, (patch_tokens, class_token, labels) in tqdm(enumerate(testloader), total=len(testloader), desc='Evaluating Test Set'):
        patch_tokens = patch_tokens.to(device)  
        class_token = class_token.to(device)    
        labels = labels.to(device)             

        outputs = model((patch_tokens, class_token))  
        _, predicted = torch.max(outputs, 1) 

        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

accuracy = correct_predictions / total_samples
print(f'Test Accuracy: {accuracy:.4f}')
