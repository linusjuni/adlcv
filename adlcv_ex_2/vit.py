import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def positional_encoding_2d(nph, npw, dim, temperature=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(nph), torch.arange(npw), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.k_projection  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_projeciton  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size, seq_len, embed_dim = x.size()
        keys    = self.k_projection(x)
        queries = self.q_projection(x)
        values  = self.v_projeciton(x)

        # Rearrange keys, queries and values 
        # from batch_size x seq_len x embed_dim to (batch_size x num_head) x seq_len x head_dim
        keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        values = rearrange(values, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)

        attention_logits = torch.matmul(queries, keys.transpose(1, 2))
        attention_logits = attention_logits * self.scale
        attention = F.softmax(attention_logits, dim=-1)
        out = torch.matmul(attention, values)

        # Rearragne output
        # from (batch_size x num_head) x seq_len x head_dim to batch_size x seq_len x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_len, seq_len)
        assert out.size() == (batch_size, seq_len, embed_dim)

        return self.o_projection(out)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.GELU(),
            nn.Linear(fc_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.layernorm1(attention_out + x)
        x = self.dropout(x)
        fc_out = self.fc(x)
        x = self.layernorm2(fc_out + x)
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, image_size, channels, patch_size, embed_dim, num_heads, num_layers,
                 pos_enc='fixed', pool='cls', dropout=0.0, 
                 fc_dim=None, num_classes=2, ):
        
        super().__init__()

        assert pool in ['cls', 'mean', 'max']
        assert pos_enc in ['fixed', 'learnable']

        self.pool, self.pos_enc, = pool, pos_enc

        H, W = image_size
        patch_h, patch_w = patch_size
        assert H % patch_h == 0 and W % patch_w == 0, 'Image dimensions must be divisible by the patch size'

        num_patches = (H // patch_h) * (W // patch_w)
        patch_dim = channels * patch_h * patch_w

        if self.pool == 'cls':
            self.cls_token = nn.Parameter(torch.rand(1,1,embed_dim))
        
        # TASK: Implement patch embedding layer 
        #       Convert imaged to patches and project to the embedding dimension
        # HINT: 1) Use the Rearrange layer from einops.layers.torch 
        #          in the same way you used the rearrange function 
        #          in the image_to_patches function (playground.py)
        #       2) Stack Rearrange layer with a linear projection layer using nn.Sequential
        #          Consider including LayerNorm layers before and after the linear projection

        # ============================================
        # PATCH EMBEDDING LAYER
        # ============================================
        # Converts image to sequence of patch embeddings
        # Input:  (B, 3, 32, 32)  - batch of RGB images
        # Output: (B, 64, embed_dim) - batch of embedded patches
        
        self.to_patch_embedding = nn.Sequential(
            # Step 1: Rearrange image into flattened patches
            # (B, C, H, W) -> (B, num_patches, patch_dim)
            # Example: (B, 3, 32, 32) -> (B, 64, 48)
            # 'nph' = number of patches in height dimension (inferred as H/ph = 32/4 = 8)
            # 'npw' = number of patches in width dimension (inferred as W/pw = 32/4 = 8)
            # '(nph npw)' = combine into sequence dimension (8*8 = 64)
            # '(ph pw c)' = flatten each patch (4*4*3 = 48)
            Rearrange('b c (nph ph) (npw pw) -> b (nph npw) (ph pw c)', 
                      ph=patch_h, pw=patch_w),
            
            # Step 2: Normalize flattened patches
            # Normalizes across the patch dimension (48)
            # Helps stabilize training by ensuring similar scales
            nn.LayerNorm(patch_dim),
            
            # Step 3: Linear projection to embedding dimension
            # Projects each 48-dim patch vector to embed_dim (e.g., 128)
            # This is a learnable transformation
            # Output: (B, 64, 128)
            nn.Linear(patch_dim, embed_dim),
            
            # Step 4: Normalize embeddings
            # Normalizes across the embedding dimension
            # Prepares embeddings for positional encoding addition
            nn.LayerNorm(embed_dim),
        )

        # ============================================
        # CLS TOKEN (for 'cls' pooling strategy)
        # ============================================
        # Learnable classification token prepended to sequence
        # Shape: (1, 1, embed_dim) - will be expanded to batch size
        # This token aggregates information from all patches
        # After transformer, we use this token's output for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ============================================
        # POSITIONAL ENCODING
        # ============================================
        # Adds spatial position information to patches
        # Transformers have no inherent notion of order/position
        
        if self.pos_enc == 'learnable':
            # Learnable positional embeddings
            # One embedding vector for each patch position
            # Shape: (num_patches, embed_dim) or (num_patches+1, embed_dim) if using cls token
            # Initialized randomly and learned during training
            if self.pool == 'cls':
                # +1 for the CLS token position
                self.positional_embedding = nn.Parameter(
                    torch.randn(num_patches + 1, embed_dim)
                )
            else:
                self.positional_embedding = nn.Parameter(
                    torch.randn(num_patches, embed_dim)
                )
                
        elif self.pos_enc == 'fixed':
            # Fixed sinusoidal positional encodings (like original Transformer)
            # NOT learned, fixed mathematical function
            # Uses 2D sine/cosine patterns based on patch positions
            # Output shape: (num_patches, embed_dim)
            self.positional_embedding = positional_encoding_2d(
                nph = H // patch_h,   # Number of patches in height (8)
                npw = W // patch_w,   # Number of patches in width (8)
                dim = embed_dim,      # Embedding dimension
            ) 

        # ============================================
        # TRANSFORMER ENCODER BLOCKS
        # ============================================
        # Stack of transformer layers with self-attention and MLP
        # Each block processes the sequence and updates representations
        
        transformer_blocks = []
        for i in range(num_layers):
            # Add one encoder block per layer
            # Each block contains: LayerNorm -> MultiHeadAttention -> LayerNorm -> MLP
            transformer_blocks.append(
                EncoderBlock(
                    embed_dim=embed_dim,    # Dimension of embeddings
                    num_heads=num_heads,    # Number of attention heads
                    fc_dim=fc_dim,          # Hidden dimension of MLP (usually 4*embed_dim)
                    dropout=dropout         # Dropout rate for regularization
                ))
        
        # Wrap all blocks in Sequential for easy forward pass
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        
        # ============================================
        # CLASSIFICATION HEAD
        # ============================================
        # Final linear layer to predict class logits
        # Input: (B, embed_dim) - pooled representation
        # Output: (B, num_classes) - class scores
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Dropout layer applied after adding positional encodings
        self.dropout = nn.Dropout(dropout)


    def forward(self, img):

        tokens = self.to_patch_embedding(img)
        batch_size, num_patches, embed_dim = tokens.size()
        
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 e -> b 1 e', b=batch_size)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            num_patches+=1
        
        positions =  self.positional_embedding.to(img.device, dtype=img.dtype)
        if self.pos_enc == 'fixed' and self.pool=='cls':
            positions = torch.cat([torch.zeros(1, embed_dim).to(img.device), positions], dim=0)
        x = tokens + positions
        
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        
        if self.pool =='max':
            x = x.max(dim=1)[0]
        elif self.pool =='mean':
            x = x.mean(dim=1)
        elif self.pool == 'cls':
            x = x[:, 0]

        return self.classifier(x)