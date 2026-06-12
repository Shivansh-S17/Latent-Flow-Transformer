# # student_transformer.py
# """
# Student Transformer that replaces a middle block (m..n) of the teacher transformer
# with a learned latent-flow bridge (unrolled velocity model).

# Design:
# - Reuse teacher.embedding and teacher.layers for exact compatibility.
# - Early blocks: layers[0 .. m]  (i.e., produce x_m)
# - Late blocks: layers[n+1 .. end]
# - Tokenizer: maps per-point features (B,N,D) -> tokens (B,K,D)
#     - tokenizers may be cross-attention based (B,K,D) or PointNet (B,D) (we expand dims)
# - Flow bridge: uses sample_unroll from flow_model (unroll velocity_model)
# - Detokenizer: cross-attention from point-queries to token values -> per-point features
# - Final layers: run late blocks -> pooling -> classifier head

# API:
#     StudentTransformer(teacher, tokenizer, velocity_model, m_layer, n_layer, ...)

# Assumptions:
# - `teacher` is an instance of PointTransformer that exposes:
#     - teacher.embedding (embedding MLP),
#     - teacher.layers (ModuleList of transformer blocks),
#     - teacher.norm, teacher.cls_head
#     - teacher.m_layer and teacher.n_layer attributes (or pass them)
# - `tokenizer` : callable mapping (B,N,D_pt) -> (B,K,D) or (B,D)
# - `velocity_model` : the trained TokenTransformerVelocity (or similar)
# - `sample_unroll` : function to integrate from z0 -> zT
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from flow_model import sample_unroll


# class CrossAttentionDetokenizer(nn.Module):
#     """
#     Convert tokens (B, K, D) back to per-point features (B, N, D_point)
#     by attending from point queries -> tokens (keys/values).
#     We assume we have access to original point coords to generate point queries.
#     """
#     def __init__(self, token_dim, point_dim, num_heads=4):
#         super().__init__()
#         self.point_proj = nn.Linear(3, point_dim)  # project xyz to query space (optional)
#         self.token_norm = nn.LayerNorm(token_dim)
#         # MultiheadAttention expects (seq_len, batch, embed_dim) or batch_first True
#         # We'll use batch_first=True for convenience.
#         # We will implement attention where queries = per-point projected coords/features,
#         # keys/values = tokens projected to same dim.
#         self.token_to_kv = nn.Linear(token_dim, point_dim)  # project tokens to key/value space
#         self.attn = nn.MultiheadAttention(embed_dim=point_dim, num_heads=num_heads, batch_first=True)
#         self.out_proj = nn.Linear(point_dim, point_dim)

#     def forward(self, tokens, point_coords, point_feat_residual=None):
#         """
#         tokens: (B, K, D_token)
#         point_coords: (B, N, 3)
#         point_feat_residual: optional (B, N, D_point) residual to add (e.g., early features)
#         returns: (B, N, D_point)
#         """
#         B, N, _ = point_coords.shape
#         # queries: use projected coords
#         q = self.point_proj(point_coords)  # (B, N, Dq)
#         # keys/values: project tokens to kv dimension
#         kv = self.token_to_kv(self.token_norm(tokens))  # (B, K, Dq)

#         # use MHA with queries=q, keys=kv, values=kv
#         attn_out, _ = self.attn(q, kv, kv, need_weights=False)  # (B, N, Dq)
#         out = self.out_proj(attn_out)  # (B, N, Dq)

#         if point_feat_residual is not None:
#             # If residual present and dims match, add
#             if point_feat_residual.shape == out.shape:
#                 out = out + point_feat_residual
#         return out


# class StudentTransformer(nn.Module):
#     def __init__(self,
#                  teacher,
#                  tokenizer,
#                  velocity_model,
#                  m_layer,
#                  n_layer,
#                  detokenizer=None,
#                  unroll_steps=12,
#                  freeze_teacher_early=True,
#                  device=None):
#         """
#         teacher: PointTransformer instance (pretrained or not)
#         tokenizer: callable mapping (B,N,D_pt) -> (B,K,D) or (B,D)
#         velocity_model: trained velocity (u_theta) instance
#         m_layer, n_layer: indices for compression (0-indexed)
#         detokenizer: optional CrossAttentionDetokenizer; if None we construct one
#         unroll_steps: number of Euler steps when unrolling flow
#         freeze_teacher_early: if True, freeze early layers weights (optional)
#         """
#         super().__init__()
#         self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # copy reference to teacher modules (so parameter sharing or copying is possible)
#         self.embedding = teacher.embedding  # maps (B,N,in_dim) -> (B,N,feat_dim)
#         self.feat_dim = teacher.layers[0].attn.embed_dim if hasattr(teacher.layers[0].attn, 'embed_dim') else teacher.layers[0].norm1.normalized_shape[0] if hasattr(teacher.layers[0], 'norm1') else None

#         # split teacher layers into early and late stacks
#         self.m_layer = m_layer
#         self.n_layer = n_layer
#         self.teacher_layers = teacher.layers  # ModuleList
#         # early layers: 0..m_layer (inclusive)
#         self.early_layers = nn.ModuleList([self.teacher_layers[i] for i in range(0, m_layer + 1)])
#         # late layers: n_layer+1 .. end
#         self.late_layers = nn.ModuleList([self.teacher_layers[i] for i in range(n_layer + 1, len(self.teacher_layers))])

#         # keep final norm and classifier head from teacher
#         self.final_norm = teacher.norm
#         self.classifier = teacher.cls_head

#         # tokenizer and velocity model
#         self.tokenizer = tokenizer
#         self.velocity_model = velocity_model
#         self.unroll_steps = unroll_steps

#         # detokenizer: if not provided, build one that maps tokens -> per-point features
#         # choose point_dim = embedding output dim (we will infer from embedding by running a dummy if needed)
#         # We'll set token_dim based on example passed later or use teacher.embed_dim
#         token_dim = None
#         try:
#             # attempt to infer token dim by peeking at tokenizer.token_queries if exists
#             token_dim = getattr(tokenizer, 'token_queries').shape[-1]
#         except Exception:
#             # fallback to embedding output dim if possible
#             token_dim = getattr(self.embedding[-1], 'out_features', None) if isinstance(self.embedding, nn.Sequential) else None

#         point_dim = None
#         # try to infer point_dim (feature dim after embedding)
#         sample_dim = None
#         if hasattr(teacher, 'feat_dim') and teacher.feat_dim is not None:
#             point_dim = teacher.feat_dim
#         else:
#             # fallback: use embedding MLP output size
#             if hasattr(self.embedding, 'mlp') and isinstance(self.embedding.mlp, nn.Sequential):
#                 last_linear = None
#                 for module in reversed(self.embedding.mlp):
#                     if isinstance(module, nn.Linear):
#                         last_linear = module
#                         break
#                 if last_linear is not None:
#                     point_dim = last_linear.out_features

#         if detokenizer is None:
#             # default: build one with reasonable dims
#             token_dim_final = token_dim if token_dim is not None else 128
#             point_dim_final = point_dim if point_dim is not None else 128
#             self.detokenizer = CrossAttentionDetokenizer(token_dim=token_dim_final, point_dim=point_dim_final,
#                                                          num_heads=4)
#         else:
#             self.detokenizer = detokenizer

#         # option to freeze early layers (if desired)
#         if freeze_teacher_early:
#             for p in self.early_layers.parameters():
#                 p.requires_grad = False

#     def forward(self, points):
#         """
#         points: (B, N, in_channels) - input point cloud
#         returns: logits (B, num_classes)
#         """
#         B, N, C = points.shape
#         device = points.device

#         # 1) embedding
#         x = self.embedding(points)  # (B, N, D_pt)

#         # 2) early layers up to m_layer (we want the output after layer m => x_m)
#         for blk in self.early_layers:
#             x = blk(x)

#         x_m = x  # (B, N, D_pt)  -- input to compressed region

#         # 3) tokenize x_m -> z0 tokens
#         z0 = self.tokenizer(x_m)  # could be (B,K,D) or (B,D)
#         if z0.ndim == 2:
#             # pointnet-style returns (B, D) -> treat as single token
#             z0 = z0.unsqueeze(1)  # (B,1,D)
#         # ensure float/device
#         z0 = z0.to(device)

#         # 4) unroll flow: zT = sample_unroll(velocity_model, z0, steps=unroll_steps)
#         # if velocity model expects same token shape
#         zT = sample_unroll(self.velocity_model, z0, steps=self.unroll_steps, device=device)  # (B,K,D)

#         # 5) detokenize zT -> per-point features
#         # detokenizer needs point coords; pass original points and optional residual x_m
#         x_decoded = self.detokenizer(zT, points, point_feat_residual=x_m)  # (B, N, D_pt)

#         # 6) pass through late layers
#         h = x_decoded
#         for blk in self.late_layers:
#             h = blk(h)

#         # 7) final norm, pool, classifier
#         h_norm = self.final_norm(h)  # (B,N,D)
#         pooled = h_norm.mean(dim=1)  # global avg pool (B,D)
#         logits = self.classifier(pooled)  # (B, num_classes)

#         return logits

# student_transformer.py
"""
StudentTransformer: integrates teacher early/late layers with latent-flow bridge.

Usage:
    student = StudentTransformer(
        teacher=teacher_model,
        tokenizer=tokenizer,            # Tokenizer instance (has encode/decode)
        velocity_model=velocity_model,  # pretrained TokenTransformerVelocity
        m_layer=2, n_layer=5,
        detokenizer=None,               # optional CrossAttentionDetokenizer
        unroll_steps=12,
        freeze_early=False,
        freeze_flow=False,
        device=device
    )
    logits = student(points)  # points: (B, N, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_model import sample_unroll


class CrossAttentionDetokenizer(nn.Module):
    """
    Convert tokens (B, K, D_token) back to per-point features (B, N, D_point)
    by using point queries and token keys/values via cross-attention.
    - queries: per-point features (we use raw point coordinates projected)
    - keys/values: tokens projected
    Output dimension equals point_dim.
    """

    def __init__(self, token_dim=128, point_dim=128, num_heads=4):
        super().__init__()
        # Project raw point coords (x,y,z) -> query space (point_dim)
        self.point_proj = nn.Sequential(
            nn.Linear(3, point_dim),
            nn.GELU(),
            nn.Linear(point_dim, point_dim)
        )
        # Project tokens to key/value space
        self.token_kv = nn.Linear(token_dim, point_dim)
        # Multihead attention (batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=point_dim, num_heads=num_heads, batch_first=True)
        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(point_dim, point_dim),
            nn.GELU(),
            nn.Linear(point_dim, point_dim)
        )

    def forward(self, tokens, points, residual=None):
        """
        tokens: (B, K, D_token)
        points: (B, N, 3)  -- used as queries through a small projection
        residual: optional per-point features (B, N, D_point) to add as skip connection
        returns:
            per_point_features: (B, N, D_point)
        """
        # tokens -> kv, points -> q
        B, K, D_t = tokens.shape
        Bp, N, C = points.shape
        assert B == Bp

        q = self.point_proj(points)                  # (B, N, Dq)
        kv = self.token_kv(tokens)                    # (B, K, Dq)

        # MultiheadAttention expects (B, S, D) batch_first
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)  # (B, N, Dq)
        out = self.out_proj(attn_out)  # (B, N, Dq)

        if residual is not None and residual.shape == out.shape:
            out = out + residual
        return out


class StudentTransformer(nn.Module):
    def __init__(self,
                 teacher,
                 tokenizer,
                 velocity_model,
                 m_layer,
                 n_layer,
                 detokenizer=None,
                 unroll_steps: int = 12,
                 freeze_early: bool = False,
                 freeze_flow: bool = False,
                 device: torch.device = None):
        """
        teacher: PointTransformer instance (full model)
        tokenizer: Tokenizer instance with encode/decode methods
        velocity_model: TokenTransformerVelocity (trained or untrained)
        m_layer, n_layer: indices (0-based) of the block range to compress (compress layers m..n inclusive)
        detokenizer: CrossAttentionDetokenizer or None (will be created)
        unroll_steps: Euler integration steps for sample_unroll
        freeze_early: whether to freeze early layers parameters
        freeze_flow: whether to freeze flow model params during student training
        device: torch.device
        """
        super().__init__()
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Embed and layer stacks from teacher (share weights)
        self.embedding = teacher.embedding
        self.feat_dim = teacher.feat_dim if hasattr(teacher, 'feat_dim') else None
        self.all_layers = teacher.layers  # ModuleList

        # store indices
        self.m_layer = m_layer
        self.n_layer = n_layer
        assert 0 <= m_layer < n_layer < len(self.all_layers), "m_layer < n_layer and within range required"

        # Early and late stacks
        self.early_layers = nn.ModuleList([self.all_layers[i] for i in range(0, m_layer + 1)])  # inclusive of m_layer
        self.late_layers = nn.ModuleList([self.all_layers[i] for i in range(n_layer + 1, len(self.all_layers))])

        # Keep final norm and classifier
        self.final_norm = teacher.norm
        self.classifier = teacher.cls_head

        # Tokenizer and flow
        self.tokenizer = tokenizer
        self.velocity_model = velocity_model
        self.unroll_steps = unroll_steps

        # Detokenizer: if not provided, auto-create with sensible dims
        if detokenizer is None:
            # infer dims
            token_dim = None
            if hasattr(self.tokenizer, 'latent_tokens'):
                token_dim = self.tokenizer.latent_tokens.shape[-1]
            elif hasattr(self.tokenizer, 'latent_dim'):
                token_dim = self.tokenizer.latent_dim
            else:
                token_dim = self.feat_dim or 128
            point_dim = self.feat_dim or token_dim
            detokenizer = CrossAttentionDetokenizer(token_dim=token_dim, point_dim=point_dim, num_heads=4)
        self.detokenizer = detokenizer

        # freeze early layers optionally
        if freeze_early:
            for p in self.early_layers.parameters():
                p.requires_grad = False

        # freeze flow optionally
        if freeze_flow:
            for p in self.velocity_model.parameters():
                p.requires_grad = False
            # If tokenizer was used in flow training and we want it frozen too, freeze tokenizer
            for p in self.tokenizer.parameters():
                p.requires_grad = False

    def forward(self, points, return_latents: bool = False):
        """
        Forward pass through the student model.
        points: (B, N, 3)
        return_latents: if True, also return (z0, zT) where z0 is tokens from m_layer and zT is unrolled tokens.
        Returns:
            logits: (B, num_classes)
            optionally (z0, zT)
        """
        device = points.device
        x = self.embedding(points)  # (B, N, D)
        # early layers
        for blk in self.early_layers:
            x = blk(x)
        x_m = x  # features at layer m (B, N, D)

        # tokenize
        # Tokenizer.encode expects (B, N, D) and returns (B, K, D)
        z0 = self.tokenizer.encode(x_m) if hasattr(self.tokenizer, 'encode') else self.tokenizer(x_m)
        if z0.dim() == 2:
            z0 = z0.unsqueeze(1)
        # unroll via learned velocity model
        zT = sample_unroll(self.velocity_model, z0.to(device), steps=self.unroll_steps, device=device)

        # detokenize: produce per-point features for late stack
        x_dec = self.detokenizer(zT, points, residual=x_m)  # (B, N, D)
        h = x_dec
        # pass through late layers
        for blk in self.late_layers:
            h = blk(h)

        h = self.final_norm(h)
        pooled = h.mean(dim=1)
        logits = self.classifier(pooled)

        if return_latents:
            return logits, z0, zT
        return logits


# ---------------------------
# sanity-check snippet
# ---------------------------
if __name__ == "__main__":
    # quick random sanity check for shapes
    import torch
    from point_transformer import PointTransformer
    from tokenizer import Tokenizer
    from flow_model import TokenTransformerVelocity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 2, 256
    points = torch.randn(B, N, 3).to(device)

    # build teacher and components (random init)
    teacher = PointTransformer(in_channels=3, embed_dim=64, depth=8, num_heads=4, num_classes=40).to(device)
    tokenizer = Tokenizer(embed_dim=64, num_tokens=8).to(device)
    velocity = TokenTransformerVelocity(token_dim=64, num_heads=4, num_layers=2).to(device)

    student = StudentTransformer(teacher=teacher, tokenizer=tokenizer, velocity_model=velocity, m_layer=2, n_layer=5, unroll_steps=6, device=device).to(device)
    logits, z0, zT = student(points, return_latents=True)
    print("logits.shape:", logits.shape)
    print("z0.shape:", z0.shape)
    print("zT.shape:", zT.shape)
