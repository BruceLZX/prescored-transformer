'''
K means/lev + Original Hyper
'''
import math
import torch
from einops import rearrange

from .utils import exact_attention, exact_attention_cuda, add_self_attentions, indexing
from .angular_lsh import AngularLSH

ORTHO_MATRIX = None


class HyperAttention(torch.nn.Module):

    def __init__(self, input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=4096, cuda=False, top_k=4096, score_method = "lev",use_prescore = 1):
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.cuda = cuda
        self.lsh = AngularLSH(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))

        self.use_prescore = use_prescore
        self.topk = top_k
        self.threshold = 0
        self.sample_size = sample_size

        self.kmeans_centroids = None
        self.kmeans_call_count = 0
        self.kmeans_update_freq = 50
        self.score_method = score_method

        self.gaussian_sigma_scale = 1.3

        
    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, scale=None, causal=False, return_lse=False):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape
        scale = dim ** (-0.5) if scale is None else scale
        
        # Without causal masking
        if not causal: 
            attn, lse = self.forward_no_causal_mask(query, key, value, scale)

        # With causal masking
        else:
            if n_key <= self.min_seq_len:
                if self.cuda:
                    attn, lse = exact_attention_cuda(query, key, value, scale, causal=True)
                else:
                    attn, lse = exact_attention(query, key, value, scale, causal=True)
            else:
            
                # If n_query is odd we pad inputs by adding all-zero rows
                if n_query % 2:
                    query = torch.nn.functional.pad(query, (0,0,0,1), mode='constant',value=0.)
                    key = torch.nn.functional.pad(key, (0,0,0,1), mode='constant',value=0.)
                    value = torch.nn.functional.pad(value, (0,0,0,1), mode='constant',value=0.)

                q_bd = query.view(batch_size, 2*n_heads, query.shape[2]//2, query.shape[-1])
                k_bd = key.view(batch_size, 2*n_heads, key.shape[2]//2, key.shape[-1])
                v_bd = value.view(batch_size, 2*n_heads, key.shape[2]//2, value.shape[-1])
        
                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)
                
                if attn_bd.shape[2] not in attn_bd.stride():
                    attn_bd = attn_bd.contiguous()
                attn_bd = attn_bd.view(batch_size, n_heads, -1, dim)

                if lse_bd.shape[2] not in lse_bd.stride():
                    lse_bd = lse_bd.contiguous()
                lse_bd = lse_bd.view(batch_size, n_heads, -1, 1)

                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2]//2:, :],
                    key[:, :, :key.shape[2]//2, :], 
                    value[:, :, :key.shape[2]//2, :], scale)

                attn_up, lse_up = attn_bd[:,:,:query.shape[2]//2,:], lse_bd[:,:,:query.shape[2]//2,:]
                attn_down, lse_down = add_self_attentions(
                    attn_bd[:,:,query.shape[2]//2:,:],
                    lse_bd[:,:,query.shape[2]//2:,:],
                    attn_unmasked,
                    lse_unmasked)

                attn = torch.cat((attn_up, attn_down), dim=-2)
                lse = torch.cat((lse_up, lse_down), dim=-2)

                # If n_query was odd exclude the last rows
                if n_query % 2:
                    attn = attn[:,:,:-1,:]
                    lse = lse[:,:,:-1,:]

        if not return_lse:
            return attn
        else:
            return attn, lse


    def forward_no_causal_mask(self, query, key, value, scale):

        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]
        
        if self.min_seq_len > n_query:
            if self.cuda:
                return exact_attention_cuda(query, key, value, scale, causal=False)
            else:
                return exact_attention(query, key, value, scale, causal=False)

        if self.use_prescore == 1:
            if self.score_method == "kmeans":
                scores = self._compute_kmeans_scores(query, key)
            elif self.score_method == "lev":
                scores = self._compute_lev_scores(key)
            elif self.score_method == "kmedian":
                scores = self._compute_kmedian_scores(query, key)
            elif self.score_method == "kmeans_kernel":
                scores = self._compute_kmeans_kernel_scores(key)
            elif self.score_method == "gaussian_kernel":
                scores = self._compute_gaussian_kernel_scores(key)
            else:
                raise ValueError(f"Unknown score_method {self.score_method}")

            # ---------------- selection ----------------
            if self.threshold > 0.0:
                mask = (scores >= self.threshold)
            else:
                top_k = min(self.topk, n_key)
                top_vals, top_idx = torch.topk(scores, k=top_k, dim=-1)
                mask = torch.zeros_like(scores, dtype=torch.bool)
                for b_i in range(batch_size):
                    for hh_i in range(head_size):
                        mask[b_i, hh_i, top_idx[b_i, hh_i]] = True

            key = torch.where(mask.unsqueeze(-1), key, torch.zeros_like(key))
            value = torch.where(mask.unsqueeze(-1), value, torch.zeros_like(value))

            # ensure dtype/device consistency with LSH projections
            proj_dtype = self.lsh.proj_dir.dtype
            proj_device = self.lsh.proj_dir.device
            key = key.to(device=proj_device, dtype=proj_dtype)
            value = value.to(device=proj_device, dtype=proj_dtype)
        
        # 1. Sorted block-diagonal via sortLSH
        _, query_sort_idx = torch.sort(self.lsh.hash(query), dim=2, stable=True) # batch_size x head_size x n
        _, key_sort_idx = torch.sort(self.lsh.hash(key), dim=2, stable=True)
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True) # for recovering the row order

        key_block_size = self.block_size

        query_sorted = indexing(query, query_sort_idx, key_block_size)
        key_sorted = indexing(key, key_sort_idx, key_block_size)
        value_sorted = indexing(value, key_sort_idx, key_block_size)

        if key_block_size > 0:

            num_blocks = key_sorted.shape[2] // key_block_size
            query_block_size = query_sorted.shape[2] // num_blocks

            # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
            query_split_per_block = query_sorted.view(-1, 1, query_block_size, dim)
            key_split_per_block = key_sorted.view(-1, 1, key_block_size, dim)
            value_split_per_block = value_sorted.view(-1, 1, key_block_size, dim)

            if self.cuda:
                attn_block, lse_block = exact_attention_cuda(
                    query_split_per_block, key_split_per_block, value_split_per_block,
                    softmax_scale=scale, causal=False)
            else:
                attn_block, lse_block = exact_attention(
                    query_split_per_block, key_split_per_block, value_split_per_block,
                    softmax_scale=scale, causal=False)

            if attn_block.shape[2] not in attn_block.stride():
                attn_block = attn_block.contiguous()
            attn_block = attn_block.view(batch_size, head_size, query_sorted.shape[2], -1)

            if lse_block.shape[2] not in lse_block.stride():
                lse_block = lse_block.contiguous()
            lse_block = lse_block.view(batch_size, head_size, query_sorted.shape[2], -1)

            # When inputs are padded, then unpad them
            if query_sorted.shape[2] != n_query: #query.shape[2]:
                attn_block, lse_block = attn_block[:,:,:n_query,:], lse_block[:,:,:n_query,:]
                query_sorted = query_sorted[:,:,:n_query,:]
                key_sorted = key_sorted[:,:,:n_key,:]
                value_sorted = value_sorted[:,:,:n_key,:]

        else:
            query_block_size = -1
            query_block_size = -1
            attn_block, lse_block = 0, 0

        # 2. Residual low-rank part via uniform sampling
        # Sample indices uniformly at random
        sample_size = self.sample_size
        if sample_size > 0 and (n_query > query_block_size) and (n_key > key_block_size):
            sampled_set = torch.randint(n_key, size=(batch_size, head_size, sample_size), device=query_sorted.device)
            
            # Compute mask for hiding A_ij computed in block-diagonal attention
            offset_n = rearrange(torch.arange(n_query, device=query_sorted.device), 'n -> 1 n 1')
            weights = n_key / sample_size
            value_subset = indexing(value_sorted, sampled_set)
            key_subset = indexing(key_sorted, sampled_set)

            if not self.cuda:
                block_mask = (offset_n // query_block_size) == (sampled_set // key_block_size).view(-1, 1, sample_size)
                block_mask = block_mask.view(batch_size, head_size, -1, sample_size)
                block_mask = block_mask.to(query_sorted.dtype)
                block_mask *= torch.finfo(query_sorted.dtype).min # adding -inf added to QK^T

                attn_res, lse_res = exact_attention(query_sorted, key_subset, value_subset, scale, causal=False, bias=block_mask)
            else:
                attn_res, lse_res = exact_attention_cuda(query_sorted, key_subset, value_subset, scale, causal=False)
            lse_res = lse_res + math.log(weights)

            # Add two attentions
            if key_block_size > 0:
                attn, lse = add_self_attentions(attn_block, lse_block, attn_res, lse_res)
            else:
                attn, lse = attn_res, lse_res
        else:
            attn, lse = attn_block, lse_block

        # Re-order rows with the inverse order for query_sorted -> query
        attn = indexing(attn, query_sort_idx_inv)
        lse = indexing(lse, query_sort_idx_inv)
        return attn, lse

    def _compute_kmeans_scores(self, query, key):
        bsz, hsz, n_key, dim = key.shape
        device = key.device

        self.kmeans_call_count += 1
        do_update = False
        if (self.kmeans_centroids is None) or (self.kmeans_call_count % self.kmeans_update_freq == 1):
            do_update = True

        keys_2d = key.view(bsz * hsz, n_key, dim).float()

        if do_update:
            #k_clusters = max(1, min(self.topk // 4, n_key))
            k_clusters = max(1, min(256, self.topk // 4))
            k_iters = 10
            centroids = self._big_kmeans(keys_2d, k_clusters, k_iters)
            self.kmeans_centroids = centroids
        else:
            centroids = self.kmeans_centroids

        dist = torch.cdist(keys_2d, centroids, p=2)  # [B*H, N, k]
        assign = dist.argmin(dim=-1)  # [B*H, N]
        dist_min = dist.gather(2, assign.unsqueeze(-1)).squeeze(-1)  # [B*H, N]

        # Vectorized way to calculate the distance and
        k = centroids.shape[0]
        dist_flat = dist_min.view(-1)              # [B*H*N]
        assign_flat = assign.view(-1)              # [B*H*N]
        cluster_sum = torch.zeros(k, device=device).scatter_add(0, assign_flat, dist_flat)
        scores = dist_flat / (cluster_sum[assign_flat] + 1e-6)
        scores = scores.view(bsz, hsz, n_key)
        return scores

    def _big_kmeans(self, data_3d, k_clusters, k_iters):
        device = data_3d.device
        BH, N, D = data_3d.shape
        data_2d = data_3d.reshape(BH*N, D)

        if k_clusters >= BH*N:
            return data_2d[torch.randperm(BH*N)[:k_clusters]]

        centroids = self._kmeans_plus_plus_init(data_2d, k_clusters)
        for _ in range(k_iters):
            dist = torch.cdist(data_2d, centroids, p=2)
            assign = dist.argmin(dim=-1)
            for c in range(k_clusters):
                mask = (assign == c)
                if mask.any():
                    centroids[c] = data_2d[mask].mean(dim=0)
        return centroids

    def _kmeans_plus_plus_init(self, data, k):
        X, D = data.shape
        device = data.device
        if k >= X:
            return data[torch.randperm(X)[:k]]

        centroids = torch.zeros(k, D, device=device)
        idx = torch.randint(0, X, (1,))
        centroids[0] = data[idx]

        for i in range(1, k):
            dist = torch.cdist(data, centroids[:i], p=2).min(dim=-1).values
            prob = dist / dist.sum()
            nxt  = torch.multinomial(prob, 1)
            centroids[i] = data[nxt]
        return centroids

    # ============== LevScore 计算函数 ==============
    def _compute_lev_scores(self, key: torch.Tensor) -> torch.Tensor:
        """
        Project the key (ortho), and take the L2 norm^2 as "LevScore".
        key: [bsz, hsz, n_key, dim]
        return: [bsz, hsz, n_key]
        """
        # (1) ortho投影
        orth_key = ortho_pytorch(key)
        # (2) 范数^2
        lev_scores = torch.norm(orth_key, p=2, dim=-1).pow(2)
        return lev_scores


    # -------------------- k-means(已换成 k-median) + 缓存 --------------------
    def _compute_kmedian_scores(self, query, key):
        """
        合并(B,H) 做一次 k-median => dist => 1/(min_dist+1e-6)
        """
        bsz, hsz, n_key, dim = key.shape
        device = key.device

        self.kmeans_call_count += 1
        do_update = False
        if (self.kmeans_centroids is None) or (self.kmeans_call_count % self.kmeans_update_freq == 1):
            do_update = True

        # 合并
        keys_2d = key.view(bsz * hsz, n_key, dim).float()

        if do_update:
            k_clusters = max(4, min(self.topk // 4, 256))
            k_iters = 3
            centroids = self._big_kmedian(keys_2d, k_clusters, k_iters)
            self.kmeans_centroids = centroids
        else:
            centroids = self.kmeans_centroids  # 缓存

        # 计算距离
        dist = torch.cdist(keys_2d, centroids, p=2)  # [B*H, N, k]
        assign = dist.argmin(dim=-1)  # [B*H, N]
        dist_min = dist.gather(2, assign.unsqueeze(-1)).squeeze(-1)  # [B*H, N]

       # 向量化方式计算每个 cluster 的距离和
        k = centroids.shape[0]
        dist_flat = dist_min.view(-1)              # [B*H*N]
        assign_flat = assign.view(-1)              # [B*H*N]
        cluster_sum = torch.zeros(k, device=device).scatter_add(0, assign_flat, dist_flat)
        scores = dist_flat / (cluster_sum[assign_flat] + 1e-6)
        scores = scores.view(bsz, hsz, n_key)
        return scores
        '''dist = torch.cdist(keys_2d, centroids, p=2)  # [B*H, N, k_clusters]
        min_dist, _ = dist.min(dim=-1)  # [B*H, N]
        scores_2d = 1.0 / (min_dist + 1e-6)
        scores = scores_2d.view(bsz, hsz, n_key)
        return scores'''

    def _big_kmedian(self, data_3d, k_clusters, k_iters):
        """
        data_3d: [B*H, N, D]
        flatten => [B*H*N, D] => k-median => [k_clusters, D]
        Note: k-means++ initializer remains, iteration change to 'median' from 'mean'
        """
        device = data_3d.device
        BH, N, D = data_3d.shape
        data_2d = data_3d.reshape(BH*N, D)

        if k_clusters >= BH*N:
            return data_2d[torch.randperm(BH*N)[:k_clusters]]

        # 1) k-means++ init (未变)
        centroids = self._kmeans_plus_plus_init(data_2d, k_clusters)

        # 2) 迭代: 计算中位数
        for _ in range(k_iters):
            dist = torch.cdist(data_2d, centroids, p=2)
            assign = dist.argmin(dim=-1)
            for c in range(k_clusters):
                mask = (assign == c)
                if mask.any():
                    # === 把 'mean' 换成 'median' ===
                    # PyTorch: median(...) -> (values, indices), 只取 values
                    values, _ = data_2d[mask].median(dim=0)
                    centroids[c] = values
        return centroids


    def _compute_kmeans_kernel_scores(self, key: torch.Tensor) -> torch.Tensor:
        """Kernelised variant of K‑Means pre‑scoring.

        For each key token we compute its squared Euclidean distance to the nearest centroid and
        map it through a Gaussian/RBF kernel exp(-d^2 / (2σ^2)).  σ is estimated on‑the‑fly as the
        mean of the minimum distances and scaled by ``self.gaussian_sigma_scale``.
        """
        bsz, hsz, n_key, dim = key.shape
        device = key.device

        # refresh or reuse centroids
        self.kmeans_call_count += 1
        do_update = (self.kmeans_centroids is None) or (self.kmeans_call_count % self.kmeans_update_freq == 1)

        keys_2d = key.view(bsz * hsz, n_key, dim).float()
        if do_update:
            k_clusters = max(1, min(256, self.topk // 4))
            centroids = self._big_kmeans(keys_2d, k_clusters, k_iters=10)
            self.kmeans_centroids = centroids
        else:
            centroids = self.kmeans_centroids

        # pairwise distances and assignment
        dist = torch.cdist(keys_2d, centroids, p=2)  # [B*H, N, k]
        min_dist, _ = dist.min(dim=-1)               # [B*H, N]

        # estimate σ and compute RBF kernel value
        sigma = (min_dist.mean(dim=-1, keepdim=True) * self.gaussian_sigma_scale).clamp(min=1e-6)
        kernel_vals = torch.exp(- (min_dist ** 2) / (2 * sigma ** 2))  # [B*H, N]

        scores = kernel_vals.view(bsz, hsz, n_key)
        return scores

    def _compute_gaussian_kernel_scores(self, key: torch.Tensor) -> torch.Tensor:
        """Gaussian kernel scores using the vector norm of each key.

        Score_i = exp(-||k_i||^2 / (2σ^2)) where σ is data‑driven (mean of squared norms) times
        ``self.gaussian_sigma_scale``.  This is a simple content‑based saliency signal that favours
        low‑energy keys (or high‑energy ones if you flip the inequality in top‑k selection).
        """
        # compute squared L2 norms in float32 to avoid precision loss
        norm_sq = torch.norm(key.float(), dim=-1).pow(2)   # [B, H, N]
        sigma = (norm_sq.mean(dim=(-2, -1), keepdim=True) * self.gaussian_sigma_scale).clamp(min=1e-6)
        scores = torch.exp(- norm_sq / (2 * sigma ** 2))
        return scores


def ortho_pytorch(x: torch.Tensor) -> torch.Tensor:
    """
    Multiply the last dimension of x by a random orthogonal matrix W.
    Compatible with bfloat16: Generate with float32 first, do QR, and then convert back to x.dtype.

    x.shape = [..., d]
    return shape is same as x 
    """
    global ORTHO_MATRIX
    d = x.shape[-1]

    # First time or d changed => regenerate
    if ORTHO_MATRIX is None or ORTHO_MATRIX.shape != (d, d) or ORTHO_MATRIX.device != x.device:
        # 1) firstly use float32 to generate random matrix
        mat = torch.randn(d, d, device=x.device, dtype=torch.float32)

        # 2) QR Decomposition
        q, r = torch.linalg.qr(mat, mode='reduced')  # q: [d, d], float32

        # 3) Convert back to x.dtype (may be bfloat16, float16, float32, etc.)
        q = q.to(x.dtype)

        ORTHO_MATRIX = q  # [d, d], On the same device as x

    # Multiply the last dimension of x by ORTHO_MATRIX
    orig_shape = x.shape        # [..., d]
    x_2d = x.reshape(-1, d)     # [*, d]

    # Matrix multiplication may also be numerically lossy if x is bf16/half.
    # To avoid further errors, you can upscale to float32 and then downgrade back to x.dtype:
    x_2d_float = x_2d.to(torch.float32)
    W_float     = ORTHO_MATRIX.to(torch.float32)

    x_proj_2d = x_2d_float @ W_float
    # turn to original dtype
    x_proj_2d = x_proj_2d.to(x.dtype)

    x_proj = x_proj_2d.view(orig_shape)
    return x_proj

