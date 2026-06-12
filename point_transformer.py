# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ---------- Utility: Point Feature Embedding ----------
# class PointEmbedding(nn.Module):
#     def __init__(self, in_channels=3, embed_dim=64):
#         super(PointEmbedding, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim)
#         )

#     def forward(self, x):
#         # x: (B, N, in_channels)
#         return self.mlp(x)  # (B, N, embed_dim)


# # ---------- Core Transformer Block ----------
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
#         super(TransformerBlock, self).__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)

#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(dim * mlp_ratio), dim)
#         )

#     def forward(self, x):
#         # x: (B, N, D)
#         attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
#         x = x + attn_out  # residual
#         x = x + self.mlp(self.norm2(x))
#         return x


# # ---------- Full Point Transformer Encoder ----------
# class PointTransformer(nn.Module):
#     def __init__(self, in_channels=3, embed_dim=64, depth=8, num_heads=4, num_classes=40, mlp_ratio=2.0):
#         super(PointTransformer, self).__init__()

#         self.embedding = PointEmbedding(in_channels, embed_dim)
#         self.layers = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads, mlp_ratio)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)
#         self.cls_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, num_classes)
#         )

#     def forward(self, x, return_layers=None):
#         """
#         x: (B, N, 3)
#         return_layers: list of layer indices to extract (e.g. [3, 5])
#         """
#         h = self.embedding(x)
#         hidden_states = {}

#         for i, layer in enumerate(self.layers):
#             h = layer(h)
#             if return_layers and i in return_layers:
#                 hidden_states[i] = h.clone()

#         h = self.norm(h)
#         # Global average pooling for classification
#         h_cls = h.mean(dim=1)
#         logits = self.cls_head(h_cls)

#         if return_layers:
#             return logits, hidden_states
#         else:
#             return logits

# point_transformer.py
import torch
import torch.nn as nn

class PointEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class PointTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, depth=8, num_heads=4, num_classes=40, mlp_ratio=2.0):
        super().__init__()
        self.embedding = PointEmbedding(in_channels, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, num_classes))
        self.feat_dim = embed_dim
        self.m_layer = None
        self.n_layer = None

    def forward(self, x, return_layers=None):
        # x: (B, N, 3)
        h = self.embedding(x)  # (B, N, D)
        hidden = {}
        for i, blk in enumerate(self.layers):
            h = blk(h)
            if return_layers and i in return_layers:
                hidden[i] = h.clone()
        h = self.norm(h)
        pooled = h.mean(dim=1)
        logits = self.cls_head(pooled)
        if return_layers:
            return logits, hidden
        return logits
