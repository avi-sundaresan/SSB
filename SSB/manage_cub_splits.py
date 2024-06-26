import os
import random 
from utils import load_class_splits, load_config
import sys
import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import math
import random
import PIL
from PIL import Image
from torch import nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score

config = load_config()
root = config['cub_directory']
random.seed(10)

known = load_class_splits('cub')['known_classes']
easy_unknown = load_class_splits('cub')['unknown_classes']['Easy']
medium_unknown = load_class_splits('cub')['unknown_classes']['Medium']
hard_unknown = load_class_splits('cub')['unknown_classes']['Hard']

ins = []
easy_oos = []
med_oos = []
hard_oos = []

dr = '/home/avisund/data/CUB/CUB_200_2011/images/'
folder_names = os.listdir(dr)

for i in range(len(folder_names)):
    if '.DS_Store' in folder_names[i]:
      continue
    if i in known:
        ins.append(folder_names[i])
    elif i in easy_unknown:
        easy_oos.append(folder_names[i])
    elif i in medium_unknown:
        med_oos.append(folder_names[i])
    elif i in hard_unknown:
        hard_oos.append(folder_names[i])

train_list = []
test_list = []
train_labels = []
test_labels = []
label_count = 0

label_map = {}

for i in range(len(ins)):
    imgs = os.listdir(dr + ins[i])
    test = random.sample(imgs, int(0.2 * len(imgs)))
    train_list += [dr + ins[i] + '/' + elem for elem in imgs if elem not in test]

    label = int(ins[i].split('.')[0]) - 1
    train_labels += (len(imgs) - int(0.2 * len(imgs))) * [label_count]
    test_list += [dr + ins[i] + '/' + elem for elem in test]
    test_labels += int(0.2 * len(imgs)) * [label_count]
    label_map[label] = label_count
    label_count += 1

easy_oos_list = []
med_oos_list = []
hard_oos_list = []
easy_oos_labels = []
med_oos_labels = []
hard_oos_labels = []
for elem in easy_oos:
    easy_oos_list += [dr + elem + '/' + e for e in os.listdir(dr + elem)]
    label = int(elem.split('.')[0]) - 1
    easy_oos_labels += [label_count] * len(os.listdir(dr + elem))
    label_map[label] = label_count
    label_count += 1

for elem in med_oos:
    med_oos_list += [dr + elem + '/' + e for e in os.listdir(dr + elem)]
    label = int(elem.split('.')[0]) - 1
    med_oos_labels += [label_count] * len(os.listdir(dr + elem))
    label_map[label] = label_count
    label_count += 1

for elem in hard_oos:
    hard_oos_list += [dr + elem + '/' + e for e in os.listdir(dr + elem)]
    label = int(elem.split('.')[0]) - 1
    hard_oos_labels += [label_count] * len(os.listdir(dr + elem))
    label_map[label] = label_count
    label_count += 1


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert('RGB')

        label = label_map[int(image_filepath.split('/')[-2].split('.')[0]) - 1]
        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                print(image_filepath)

        return image, label

transformation = transforms.Compose([
        transforms.Resize((224, 224), interpolation= PIL.Image.Resampling.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet default mean and sd
    ])

train_dataset = CustomDataset(train_list, transformation)
test_dataset = CustomDataset(test_list, transformation)
easy_otest_dataset = CustomDataset(easy_oos_list, transformation)
med_otest_dataset = CustomDataset(med_oos_list, transformation)
hard_otest_dataset = CustomDataset(hard_oos_list, transformation)

batch_size = 8
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

easy_opentestloader = torch.utils.data.DataLoader(easy_otest_dataset, batch_size=batch_size, shuffle=True)
med_opentestloader = torch.utils.data.DataLoader(med_otest_dataset, batch_size=batch_size, shuffle=True)
hard_opentestloader = torch.utils.data.DataLoader(hard_otest_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', skip_validation=True).to(device)

from functools import partial

class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features
n_last_blocks = 1
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)
feature_model = ModelWithIntermediateLayers(dinov2_vitg14, n_last_blocks, autocast_ctx).to("cuda")

def precompute_embeddings(dataloader, feature_model):
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

train_embeddings, train_labels = precompute_embeddings(trainloader, feature_model)
val_embeddings, val_labels = precompute_embeddings(valloader, feature_model)

from torch.utils.data import TensorDataset, DataLoader

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

train_embeddings_dataset = EmbeddingsDataset(train_embeddings, train_labels)
val_embeddings_dataset = EmbeddingsDataset(val_embeddings, val_labels)

trainloader = DataLoader(train_embeddings_dataset, batch_size=None, shuffle=True)
valloader = DataLoader(val_embeddings_dataset, batch_size=None, shuffle=True)

sys.path.append(os.path.expanduser('~/models'))

# Now you can import the jepa module
import jepa

from jepa.src.models.utils.modules import (
    Block,
    CrossAttention,
    CrossAttentionBlock
)
from jepa.src.utils.tensors import trunc_normal_

class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer)
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer)
                for i in range(depth-1)])

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q


class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )
        self.linear = nn.Linear(2 * embed_dim, num_classes, bias=True)

    def forward(self, x):
        (patch_tokens, class_token) = x
        pooled_output = self.pooler(patch_tokens).squeeze(1)
        combined_output = torch.cat([pooled_output, class_token], dim=-1)
        x = self.linear(combined_output)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )
        self.linear = nn.Linear(2 * embed_dim, num_classes, bias=True)

    def forward(self, x):
        (patch_tokens, class_token) = x
        pooled_output = self.pooler(patch_tokens).squeeze(1)
        combined_output = torch.cat([pooled_output, class_token], dim=-1)
        x = self.linear(combined_output)
        return x

def train_and_validate(lr, accum_iter, num_epochs):
    model = AttentiveClassifier(embed_dim=1536, num_classes=10000).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_train_epoch = -1
    best_val_epoch = -1

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

        avg_train_loss = train_loss / len(trainloader)

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_train_epoch = e

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (patch_tokens, class_token, labels) in tqdm(enumerate(valloader), total=len(valloader)):
                patch_tokens = patch_tokens.to(device)
                class_token = class_token.to(device)
                labels = labels.to(device)

                outputs = model((patch_tokens, class_token))
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(valloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_epoch = e

    return best_train_loss, best_val_loss, best_train_epoch, best_val_epoch

# Example grid search
learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
accum_iters = [1, 4, 8, 16, 32]
num_epochs = 20

for lr in learning_rates:
    for accum_iter in accum_iters:
        print(f'Training with lr={lr}, accum_iter={accum_iter}')
        best_train_loss, best_val_loss, best_train_epoch, best_val_epoch = train_and_validate(lr, accum_iter, num_epochs)
        with open('attentive_grid_search_results.txt', 'a') as f:
            f.write(f'Learning Rate: {lr}, Accum Iter: {accum_iter}, Best Train Loss: {best_train_loss:.4f}, '
                    f'Best Val Loss: {best_val_loss:.4f}, Best Train Epoch: {best_train_epoch}, Best Val Epoch: {best_val_epoch}\n')
            
# def train_and_validate(lr, accum_iter, num_epochs):
#     model = AttentiveClassifier(embed_dim=1536, num_classes=10000).cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     best_train_loss = float('inf')
#     best_val_loss = float('inf')
#     best_train_epoch = -1
#     best_val_epoch = -1

#     for e in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         optimizer.zero_grad()

#         for batch_idx, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader)):
#             images = images.to(device)
#             labels = labels.to(device)

#             with torch.inference_mode():
#                 features = feature_model(images)

#             ((patch_tokens, class_token),) = features
#             patch_tokens = patch_tokens.clone()
#             class_token = class_token.clone()

#             outputs = model((patch_tokens, class_token))
#             loss = criterion(outputs, labels)
#             loss.backward()

#             train_loss += loss.item()

#             if (batch_idx + 1) % accum_iter == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()

#         avg_train_loss = train_loss / len(trainloader)

#         if avg_train_loss < best_train_loss:
#             best_train_loss = avg_train_loss
#             best_train_epoch = e

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch_idx, (images, labels) in tqdm(enumerate(valloader), total=len(valloader)):
#                 images = images.to(device)
#                 labels = labels.to(device)

#                 with torch.inference_mode():
#                     features = feature_model(images)

#                 ((patch_tokens, class_token),) = features

#                 outputs = model((patch_tokens, class_token))
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item()

#         avg_val_loss = val_loss / len(valloader)

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_val_epoch = e

#     return best_train_loss, best_val_loss, best_train_epoch, best_val_epoch

# # Example grid search
# learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3] 
# accum_iters = [1, 4, 8, 16, 32] 
# num_epochs = 20

# for lr in learning_rates:
#     for accum_iter in accum_iters:
#         print(f'Training with lr={lr}, accum_iter={accum_iter}')
#         best_train_loss, best_val_loss, best_train_epoch, best_val_epoch = train_and_validate(lr, accum_iter, num_epochs)
#         with open('attentive_grid_search_results.txt', 'a') as f:
#             f.write(f'Learning Rate: {lr}, Accum Iter: {accum_iter}, Best Train Loss: {best_train_loss:.4f}, '
#                     f'Best Val Loss: {best_val_loss:.4f}, Best Train Epoch:
#                     {best_train_epoch}, Best Val Epoch: {best_val_epoch}\n')
