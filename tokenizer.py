import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer(nn.Module):
    """
    Tokenizer module for point cloud latent compression.

    It compresses the full point cloud embedding (B, N, C)
    into a smaller set of latent tokens (B, K, C), and can
    reconstruct back to the original dimensionality.

    Arguments:
    ----------
    embed_dim : int
        Feature dimension (C)
    num_tokens : int
        Number of latent tokens (K)
    """

    def __init__(self, embed_dim=64, num_tokens=8):
        super(Tokenizer, self).__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Learnable latent tokens (like CLS / register tokens)
        self.latent_tokens = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        # Encoder: project per-point features -> latent space
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Cross-attention to pool point features into K tokens
        self.attn_pool = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Decoder: expand latent tokens back to per-point embeddings
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def encode(self, x):
        """
        Encode full feature map (B, N, C) into latent tokens (B, K, C)
        """
        B, N, C = x.shape
        x_proj = self.encoder(x)  # (B, N, C)

        # Expand latent tokens across batch
        latent_tokens = self.latent_tokens.expand(B, -1, -1)  # (B, K, C)

        # Cross-attend: latent tokens query the projected features
        z, _ = self.attn_pool(latent_tokens, x_proj, x_proj)  # (B, K, C)
        return z

    def decode(self, z, N):
        """
        Decode latent tokens back to per-point representation (B, N, C)
        """
        B, K, C = z.shape
        # Expand tokens to N points (broadcast)
        z_expanded = z.unsqueeze(2).expand(B, K, N, C).mean(dim=1)  # (B, N, C)
        x_recon = self.decoder(z_expanded)
        return x_recon

    def forward(self, x):
        """
        Forward through encoder and decoder
        Returns both latent z and reconstructed features
        """
        B, N, C = x.shape
        z = self.encode(x)
        x_recon = self.decode(z, N)
        return z, x_recon
