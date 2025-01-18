import torch
import torch.nn as nn
import timm
from torchvision import transforms

'''class TokenSelector(nn.Module):
    """
    Token selector with a single self-attention layer.
    Removes previous mean concat; each token can now attend
    to all other tokens before producing a binary mask.
    """
    def __init__(self, embed_dim, temperature=1.0, num_heads=1):
        """
        Args:
            embed_dim (int): Dimension of the patch embeddings.
            temperature (float): Temperature factor for sigmoid.
            num_heads (int): Number of attention heads to use.
        """
        super().__init__()
        self.temperature = temperature
        
        # A single self-attention block (scaled dot-product)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Linear projection from attended features -> scalar logits
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] patch embeddings (already pos-embedded).
        Returns:
            A mask of shape [B, N, 1]. Training => soft in [0,1].
                                       Inference => hard 0/1.
        """
        # Self-attention (batch_first=True, so x is [B, N, D])
        attended, _ = self.attn(x, x, x)
        
        # Residual + layer norm
        x = x + attended
        x = self.norm(x)
        
        # Project to scalar logits => shape [B, N]
        logits = self.out_proj(x).squeeze(-1)  # [B, N]
        
        # Temperature scaling and sigmoid
        probs = torch.sigmoid(logits / self.temperature)
        
        if self.training:
            return probs.unsqueeze(-1)  # [B, N, 1] (soft)
        else:
            return (probs > 0.5).float().unsqueeze(-1)  # [B, N, 1] (hard)'''

class TokenSelector(nn.Module):
    def __init__(self, embed_dim, temperature=1.0, num_heads=8):
        super().__init__()
        self.temperature = temperature
        
        # Project to larger dim for more expressivity
        self.expanded_dim = embed_dim * 2
        self.up_proj = nn.Linear(embed_dim, self.expanded_dim)
        
        # Multi-head attention in expanded space
        self.attn = nn.MultiheadAttention(
            self.expanded_dim, 
            num_heads,  # more heads since we have more dim
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.expanded_dim)
        
        # Final projection sequence to scalar
        self.out_proj = nn.Sequential(
            nn.Linear(self.expanded_dim, self.expanded_dim // 2),
            nn.GELU(),
            nn.Linear(self.expanded_dim // 2, 1)
        )

    def forward(self, x):
        # Project up to higher dim
        x = self.up_proj(x)  # [B, N, 2D]
        
        # Self-attention in expanded space
        attended, _ = self.attn(x, x, x)
        x = x + attended
        x = self.norm(x)
        
        # Project to scalar logits and apply sigmoid
        logits = self.out_proj(x).squeeze(-1)
        probs = torch.sigmoid(logits / self.temperature)
        
        if self.training:
            return probs.unsqueeze(-1)  # [B, N, 1] (soft)
        else:
            return (probs > 0.5).float().unsqueeze(-1)  # [B, N, 1] (hard)

class SparseViT(nn.Module):
    """
    A Vision Transformer that employs a token selector module for
    content-aware patch truncation, now with attention-based token selection.
    """
    def __init__(self, 
                 num_classes=1000, 
                 temperature=1.0, 
                 sparsity_target=0.5):
        super().__init__()
        
        # 1) Load a pretrained ViT from timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        embed_dim = self.vit.embed_dim  # Typically 768 for vit_base_patch16_224
        
        # 2) Create the token selector with self-attention
        #    We no longer do mean-embedding concat, so in_dim = embed_dim
        self.token_selector = TokenSelector(
            embed_dim=embed_dim,
            temperature=temperature,
            num_heads=8  # Single-headed or could use more heads if desired
        )
        
        # 3) Additional hyperparameters
        self.sparsity_target = sparsity_target
        self.num_classes = num_classes

    def get_sparsity_loss(self, masks):
        """
        L2-like penalty that encourages the average activation
        to match self.sparsity_target (currently a simple baseline).
        """
        mean_activation = masks.mean()
        return mean_activation  # Could do (mean_activation - self.sparsity_target).pow(2)
    
    def get_binary_regularization(self, masks):
        # Only apply pressure in the wider middle range (0.01-0.99)
        active_zone = (masks > 0.01) & (masks < 0.99)
        middle_dist = torch.abs(masks - 0.5)
        penalty = active_zone * (-torch.log(middle_dist + 1e-7))**2  # or **3
        return penalty.mean() * 5.0

    def forward(self, x):
        """
        Forward pass:
          1) Patchify input
          2) Concatenate CLS token + patch embeddings
          3) Add positional embeddings (once!) and apply dropout
          4) Separate CLS from patch tokens
          5) Pass patch tokens to the token selector (attn-based)
          6) Training => soft mask multiplication
             Inference => gather truncated tokens
          7) Pass resulting tokens through the ViT blocks, return logits
        """
        B = x.shape[0]
        
        # === [1] Patchify ===
        x_patches = self.vit.patch_embed(x)  # [B, N, D]
        
        # === [2] CLS token
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x_full = torch.cat((cls_token, x_patches), dim=1)  # [B, N+1, D]
        
        # === [3] Add position embeddings once + dropout
        x_full = x_full + self.vit.pos_embed[:, : x_full.size(1), :]
        x_full = self.vit.pos_drop(x_full)
        
        # Separate out CLS and patch tokens
        cls_only = x_full[:, 0:1, :]   # [B, 1, D]
        patch_only = x_full[:, 1:, :]  # [B, N, D]
        
        # === [4] Attn-based token selection
        patch_masks = self.token_selector(patch_only)  # shape [B, N, 1]

        if self.training:
            # Soft mask
            masked_patches = patch_only * patch_masks
            x_combined = torch.cat([cls_only, masked_patches], dim=1)  # [B, N+1, D]
            
            # Pass through ViT blocks
            for blk in self.vit.blocks:
                x_combined = blk(x_combined)
            
            x_combined = self.vit.norm(x_combined)
            logits = self.vit.head(x_combined[:, 0])
            
            return logits, patch_masks
        
        else:
            # Hard truncation
            x_out_list = []
            for i in range(B):
                sample_mask = patch_masks[i, :, 0]  # [N]
                selected_indices = (sample_mask > 0).nonzero(as_tuple=False).squeeze(-1)
                
                sample_patches = patch_only[i]  # [N, D] (already pos-embedded)
                selected_patches = sample_patches[selected_indices]
                
                sample_cls = cls_only[i : i+1, :].squeeze(1)  # [1, D]
                #print(sample_cls.shape)
                
                truncated_tokens = torch.cat([sample_cls, selected_patches], dim=0)
                truncated_tokens = truncated_tokens.unsqueeze(0)  # [1, num_sel+1, D]
                
                # Pass through ViT blocks
                for blk in self.vit.blocks:
                    truncated_tokens = blk(truncated_tokens)
                
                truncated_tokens = self.vit.norm(truncated_tokens)
                out_i = self.vit.head(truncated_tokens[:, 0])
                x_out_list.append(out_i)
            
            logits = torch.cat(x_out_list, dim=0)  # [B, num_classes]
            return logits

# Visualization helper
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
def visualize_masks(image, mask, patch_size=16, save_path=None):
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image
    import numpy as np
    
    if isinstance(image, torch.Tensor):
        image = inv_normalize(image.cpu())
        image = to_pil_image(image)
    
    img_np = np.array(image)
    H, W = image.size[1], image.size[0]
    grid_h, grid_w = H // patch_size, W // patch_size
    mask_grid = mask.reshape(grid_h, grid_w)
    
    mask_upsampled = torch.nn.functional.interpolate(
        mask_grid.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='nearest'
    ).squeeze().detach().cpu().numpy()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_upsampled, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.title(f'Mask (Selected: {(mask > 0.5).float().mean():.1%})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    overlay = img_np.copy()
    overlay = overlay * (0.7 + 0.3 * mask_upsampled[..., None])
    plt.imshow(overlay.astype(np.uint8))
    plt.title('Overlay')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Quick functionality test
    model = SparseViT(num_classes=1000)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        try:
            out_eval = model(dummy_input)
            print("✓ Eval pass shape:", out_eval.shape)
        except Exception as e:
            print("Error in eval pass:", e)
    
    model.train()
    try:
        out_train, masks_train = model(dummy_input)
        print("✓ Train pass shape:", out_train.shape)
        print("  Masks shape:", masks_train.shape)
    except Exception as e:
        print("Error in train pass:", e)
