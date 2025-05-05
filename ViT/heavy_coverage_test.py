import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from datasets import load_dataset
from timm.data import resolve_data_config, create_transform
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm


def batched_kmeans_sampling(keys, num_clusters, num_samples, num_iters=10):
    """
    Vectorized k-means clustering and sampling for keys.
    
    Args:
        keys: Tensor of shape (B, H, K, D)
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
    
    # Run k-means iterations.
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

    probs_flat = torch.clamp(probs_flat, min=1e-8)
    probs_flat = probs_flat / probs_flat.sum(dim=1, keepdim=True)
    probs_flat[torch.isnan(probs_flat)] = 1.0 / K

    sampled_indices_flat = torch.multinomial(probs_flat, num_samples, replacement=False)
    sampled_indices = sampled_indices_flat.view(B, H, num_samples)
    return sampled_indices

# #kmedian
# def batched_kmeans_sampling(keys, num_clusters, num_samples, num_iters=10):
#     B, H, K, D = keys.shape
#     device = keys.device

#     # initialize centroids
#     rand_idx = torch.randint(0, K, (B, H, num_clusters), device=device)
#     centroids = torch.gather(
#         keys, 2,
#         rand_idx.unsqueeze(-1).expand(B, H, num_clusters, D)
#     )

#     for _ in range(num_iters):
#         # cast to full precision for distance
#         keys_fp = keys.float()
#         centroids_fp = centroids.float()

#         # compute L2 dists in fp32, then cast back if you like
#         dists = torch.cdist(keys_fp, centroids_fp, p=2).to(keys.dtype)
#         assignments = torch.argmin(dists, dim=-1)
#         one_hot = F.one_hot(assignments, num_clusters).to(keys.dtype)
#         centroid_sums = torch.matmul(one_hot.transpose(2, 3), keys)
#         counts = one_hot.transpose(2, 3).sum(dim=-1, keepdim=True)
#         new_centroids = centroid_sums / (counts + 1e-6)
#         empty_mask = (counts < 1e-6)
#         centroids = torch.where(empty_mask, centroids, new_centroids)

#     # final sampling step
#     keys_fp = keys.float()
#     centroids_fp = centroids.float()
#     dists = torch.cdist(keys_fp, centroids_fp, p=2).to(keys.dtype)
#     assignments = torch.argmin(dists, dim=-1).unsqueeze(-1)
#     key_dists = torch.gather(dists, dim=-1, index=assignments).squeeze(-1)

#     probs = key_dists / (key_dists.sum(dim=-1, keepdim=True) + 1e-6)
#     probs_flat = probs.view(B * H, K).clamp(min=1e-8)
#     probs_flat = probs_flat / probs_flat.sum(dim=1, keepdim=True)

#     sampled_indices_flat = torch.multinomial(probs_flat, num_samples, replacement=False)
#     return sampled_indices_flat.view(B, H, num_samples)

class CustomKMeansAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 num_clusters=4, num_samples=32, num_iters=10, epsilon=0.01):
        """
        Custom attention module using vectorized k-means clustering and sampling.
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
        self.epsilon = epsilon
    def forward(self, x):
        """
        x: Input tensor of shape (B, N, C), where N is the sequence length.
        Returns:
            out: Tensor of shape (B, N, C)
            percentage_captured: Overall heavy attention capture percentage (scalar float)
            granular_overlap_ratio: Tensor of shape (B, num_heads) containing the overlap ratio for each head
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

        full_attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        full_attn = F.softmax(full_attn_logits, dim=-1)
        heavy_mask = full_attn > self.epsilon

        total_heavy_count_full = heavy_mask.sum().item()

 
        sampled_heavy_mask = heavy_mask.gather(
            dim=-1,
            index=sampled_indices.unsqueeze(2).expand(B, self.num_heads, N, self.num_samples)
        )
        total_heavy_count_sampled = sampled_heavy_mask.sum().item()
        percentage_captured = (total_heavy_count_sampled / total_heavy_count_full * 100) if total_heavy_count_full > 0 else 0

        heavy_counts = heavy_mask.sum(dim=2)
        _, topk_indices = heavy_counts.topk(k=self.num_samples, dim=-1)
        overlap_matrix = (sampled_indices.unsqueeze(-1) == topk_indices.unsqueeze(-2))  
        overlap_per_sample = overlap_matrix.any(dim=-1).float()
        overlap_count = overlap_per_sample.sum(dim=-1)
        granular_overlap_ratio = (overlap_count / self.num_samples * 100)

        # --- Compute Output ---
        out = torch.matmul(attn, top_values)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, percentage_captured, granular_overlap_ratio



# --- Replace Standard ViT Attention with Custom K-Means Attention ---

model = timm.create_model('vit_base_patch16_224', pretrained=True)
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
        num_clusters=6,
        num_samples=128,
        num_iters=10,
        epsilon=0.5
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

'''train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")'''
test_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")

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

'''train_dataset = train_dataset.map(transform_example, batched=True, batch_size=32, num_proc=1)
train_dataset.set_format(type="torch", columns=["pixel_values", "label"])'''
test_dataset = test_dataset.map(transform_example, batched=True, batch_size=16, num_proc=1)
test_dataset.set_format(type="torch", columns=["pixel_values", "label"])

BATCH_SIZE = 16
'''train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)'''
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

capture_percentages = []
granular_overlap_list = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["pixel_values"].to(device).half()

        x = model.patch_embed(images)
        cls_token = model.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) + model.pos_embed
        x = model.pos_drop(x)
        for block in model.blocks:
            _, perc, granular_overlap = block.attn(x)
            capture_percentages.append(perc)
            granular_overlap_list.extend(granular_overlap.flatten().tolist())

capture_percentages = np.array(capture_percentages)
granular_overlap_array = np.array(granular_overlap_list)

capture_mean = np.mean(capture_percentages)
capture_median = np.median(capture_percentages)
capture_max = np.max(capture_percentages)
capture_min = np.min(capture_percentages)

overlap_mean = np.mean(granular_overlap_array)
overlap_median = np.median(granular_overlap_array)
overlap_max = np.max(granular_overlap_array)
overlap_min = np.min(granular_overlap_array)

print("\n--- Overall Heavy Attention Capture Percentage Statistics ---")
print(f"Total Samples: {len(capture_percentages)}")
print(f"Mean:   {capture_mean:.2f}%")
print(f"Median: {capture_median:.2f}%")
print(f"Max:    {capture_max:.2f}%")
print(f"Min:    {capture_min:.2f}%")

print("\n--- Overall Granular Overlap Ratio Statistics (Per Head) ---")
print(f"Total Samples: {len(granular_overlap_array)}")
print(f"Mean:   {overlap_mean:.2f}%")
print(f"Median: {overlap_median:.2f}%")
print(f"Max:    {overlap_max:.2f}%")
print(f"Min:    {overlap_min:.2f}%")

first_attn = model.blocks[0].attn
print("\n--- Custom Attention Parameters ---")
print(f"Epsilon: {first_attn.epsilon}")
print(f"Num_samples: {first_attn.num_samples}")
print(f"Num_clusters: {first_attn.num_clusters}")