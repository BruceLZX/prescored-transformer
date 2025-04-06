import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, DownloadConfig
from transformers import ViTFeatureExtractor
import timm
from tqdm import tqdm
from torchvision import transforms
from timm.data import resolve_data_config, create_transform
from PIL import Image
import torch.nn.functional as F
import numpy as np


def batched_kmeans_sampling(keys, num_clusters, num_samples, num_iters=10):
    """
    Vectorized k-means clustering and sampling for keys.
    
    Args:
        keys: Tensor of shape (B, H, K, D), where:
              B = batch size, H = number of heads,
              K = number of keys, D = key dimension.
        num_clusters: Number of clusters per (B, H) instance.
        num_samples: Number of keys to sample per head.
        num_iters: Number of k-means iterations.
        
    Returns:
        sampled_indices: Tensor of shape (B, H, num_samples) with indices of sampled keys.
    """
    B, H, K, D = keys.shape
    device = keys.device
    
    rand_idx = torch.randint(0, K, (B, H, num_clusters), device=device)
    centroids = torch.gather(keys, 2, rand_idx.unsqueeze(-1).expand(B, H, num_clusters, D))
    
    for _ in range(num_iters):
        dists = torch.cdist(keys, centroids, p=2)
        assignments = torch.argmin(dists, dim=-1)
        one_hot = F.one_hot(assignments, num_clusters).to(keys.dtype)
        centroid_sums = torch.matmul(one_hot.transpose(2, 3), keys)
        counts = one_hot.transpose(2, 3).sum(dim=-1, keepdim=True)
        new_centroids = centroid_sums / (counts + 1e-6)

        empty_mask = (counts < 1e-6)
        centroids = torch.where(empty_mask, centroids, new_centroids)
    

    dists = torch.cdist(keys, centroids, p=2)
    assignments = torch.argmin(dists, dim=-1)
    assignments_unsq = assignments.unsqueeze(-1)

    key_dists = torch.gather(dists, dim=-1, index=assignments_unsq).squeeze(-1)

    probs = key_dists / (torch.sum(key_dists, dim=-1, keepdim=True) + 1e-6)
    
    probs_flat = probs.view(B * H, K)
    sampled_indices_flat = torch.multinomial(probs_flat, num_samples, replacement=False)
    sampled_indices = sampled_indices_flat.view(B, H, num_samples)
    return sampled_indices




class CustomKMeansAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 num_clusters=4, num_samples=32, num_iters=10):
        """
        Custom attention module using vectorized k-means clustering and sampling.
        
        Args:
            dim: Embedding dimension (e.g., 768).
            num_heads: Number of attention heads.
            qkv_bias: Whether to use bias in the qkv projection.
            attn_drop: Dropout probability for attention weights.
            proj_drop: Dropout probability after the output projection.
            num_clusters: Number of clusters for k-means per head.
            num_samples: Number of keys sampled per head.
            num_iters: Number of iterations for the k-means algorithm.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.num_clusters = num_clusters
        self.num_samples = num_samples
        self.num_iters = num_iters
        
    def forward(self, x):
        """
        x: Input tensor of shape (B, N, C), where N is the sequence length.
        Returns:
            Tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        sampled_indices = batched_kmeans_sampling(k, self.num_clusters, self.num_samples, num_iters=self.num_iters)
        
        top_keys = torch.gather(k, 2, sampled_indices.unsqueeze(-1).expand(B, self.num_heads, self.num_samples, self.head_dim))
        top_values = torch.gather(v, 2, sampled_indices.unsqueeze(-1).expand(B, self.num_heads, self.num_samples, self.head_dim))

        attn_logits = torch.matmul(q, top_keys.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        
        out = torch.matmul(attn, top_values)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

model = timm.create_model('vit_large_patch16_224', pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for block in model.blocks:
    old_attn = block.attn
    dim = old_attn.proj.in_features
    num_heads = old_attn.num_heads

    custom_attn = CustomKMeansAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=old_attn.attn_drop.p if hasattr(old_attn.attn_drop, 'p') else 0.0,
        proj_drop=old_attn.proj_drop.p if hasattr(old_attn.proj_drop, 'p') else 0.0,
        num_clusters=4,
        num_samples=64,
        num_iters=10

    )

    with torch.no_grad():
        custom_attn.qkv.weight.data.copy_(old_attn.qkv.weight.data)
        if old_attn.qkv.bias is not None:
            custom_attn.qkv.bias.data.copy_(old_attn.qkv.bias.data)
        custom_attn.proj.weight.data.copy_(old_attn.proj.weight.data)
        if old_attn.proj.bias is not None:
            custom_attn.proj.bias.data.copy_(old_attn.proj.bias.data)

    block.attn = custom_attn

print("Custom k-means attention layers have been replaced.")
model.to(device).half()



BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    trust_remote_code=True
)
'''
train_dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="train",
)
'''


config = resolve_data_config({}, model=model)
transform = create_transform(**config)

def transform_example(batch):
    transformed_images = []
    for img in batch["image"]:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        transformed_images.append(transform(img))
    batch["pixel_values"] = transformed_images
    return batch


#train_dataset = train_dataset.map(transform_example, batched=True,batch_size=32, num_proc=1)
#train_dataset.set_format(type="torch", columns=["pixel_values", "label"])
#train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_dataset = test_dataset.map(transform_example, batched=True,batch_size=32, num_proc=1)
test_dataset.set_format(type="torch", columns=["pixel_values", "label"])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



criterion = nn.CrossEntropyLoss()

# === Evaluation Loop ===
def eval_only(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch["pixel_values"].to(device).half()
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

test_loss, test_acc = eval_only(model, test_loader, criterion, DEVICE)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(tqdm(dataloader, desc="Training", leave=False, mininterval=1)):
        inputs = batch["pixel_values"].to(device).half()
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs = batch["pixel_values"].to(device).half()
        labels = batch["label"].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


num_epochs = 5  # Adjust as needed
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
print(f"Final Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
'''