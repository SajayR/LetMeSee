import torch
import torch.nn as nn
import timm
from torchvision import transforms

class TokenSelector(nn.Module):
    """
    A token selector that takes in patch embeddings of shape [B, N, *]
    and outputs soft or hard masks of shape [B, N, 1].
    
    Now it expects the patch embeddings to have shape [B, N, 2D]
    if we concatenate the mean vector and the patch embedding.
    """
    def __init__(self, in_dim, temperature=1.0):
        """
        Args:
            in_dim (int): Dimensionality of token selector input
                          (in our updated design, this may be 2*embed_dim 
                          if we concatenate patch embedding + mean embedding).
            temperature (float): Temperature factor for the sigmoid.
        """
        super().__init__()
        self.temperature = temperature
        self.logits = nn.Linear(in_dim, 1)  # Projects each token embedding to a scalar
    
    def forward(self, x):
        # x shape: [B, N, in_dim], in_dim may be 2D if using mean+patch concat
        logits = self.logits(x).squeeze(-1)  # [B, N]
        probs = torch.sigmoid(logits / self.temperature)  # [B, N]
        
        if self.training:
            # Return soft masks for training
            return probs.unsqueeze(-1)  # [B, N, 1]
        else:
            # Return hard (0/1) masks for inference
            return (probs > 0.5).float().unsqueeze(-1)  # [B, N, 1]


class SparseViT(nn.Module):
    """
    A Vision Transformer that employs a token selector module for
    content-aware patch truncation.

    Key fixes in this version:
      - We only apply positional embeddings once (before masking).
      - We remove all extra positional-embedding additions in inference.
    """
    def __init__(self, 
                 num_classes=1000, 
                 temperature=1.0, 
                 sparsity_target=0.5):
        super().__init__()
        
        # 1) Load a pretrained ViT from timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        embed_dim = self.vit.embed_dim  # Typically 768 for vit_base_patch16_224
        
        # 2) Create the token selector
        #    We'll pass [patch_embedding, mean_embedding] => total dim = 2*embed_dim
        self.token_selector = TokenSelector(in_dim=2 * embed_dim, 
                                            temperature=temperature)
        
        # 3) Additional hyperparameters
        self.sparsity_target = sparsity_target
        self.num_classes = num_classes

    def get_sparsity_loss(self, masks):
        """
        L2 penalty that encourages the average activation to match self.sparsity_target.
        """
        mean_activation = masks.mean()
        return mean_activation  # or (mean_activation - self.sparsity_target).pow(2)

    def get_binary_regularization(self, masks):
        """
        Additional penalty for mask values near 0.5.
        We encourage them to saturate toward 0 or 1.
        """
        middle_penalty = -torch.log(torch.abs(masks - 0.5) + 1e-7) * 3.0
        gray_zone = (masks > 0.05) & (masks < 0.95)
        gray_penalty = gray_zone.float() * 5.0
        return middle_penalty.mean() + gray_penalty.mean()

    def forward(self, x):
        """
        Forward pass:
          1) Patchify input
          2) Concatenate CLS token + patches
          3) Add positional embeddings (once!) and do dropout
          4) Separate out patch tokens from CLS
          5) Pass patch tokens to the token selector to get a mask
             - Training: multiply patch embeddings by mask
             - Inference: gather selected patches
          6) Pass the resulting tokens through ViT blocks
          7) Return logits and (optionally) masks
        """
        B = x.shape[0]
        
        # === [1] Patchify (backbone's patch_embed) ===
        x_patches = self.vit.patch_embed(x)  # [B, N, D]
        
        # === [2] Concatenate CLS token ===
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x_full = torch.cat((cls_token, x_patches), dim=1)  # [B, N+1, D]
        
        # === [3] Add positional embeddings once, apply dropout ===
        # Note: self.vit.pos_embed is shape [1, 197, D] (for N=196 + CLS=1).
        x_full = x_full + self.vit.pos_embed[:, : x_full.size(1), :]
        x_full = self.vit.pos_drop(x_full)
        
        # === Separate the CLS and patch tokens ===
        cls_only = x_full[:, 0:1, :]   # [B, 1, D]
        patch_only = x_full[:, 1:, :]  # [B, N, D] (pos-embedded patch tokens)
        
        # === [4] Token selection step ===
        mean_patch = patch_only.mean(dim=1, keepdim=True)     # [B, 1, D]
        mean_patch_broadcast = mean_patch.expand(-1, patch_only.shape[1], -1)  # [B, N, D]
        
        # Concat original patch embedding + mean patch embedding => shape [B, N, 2D]
        patch_plus_mean = torch.cat([patch_only, mean_patch_broadcast], dim=-1)
        
        # === [5] Forward through the token selector => get mask ===
        patch_masks = self.token_selector(patch_plus_mean)  # [B, N, 1]
        
        if self.training:
            # --- Training mode: soft mask multiplication ---
            masked_patches = patch_only * patch_masks  # [B, N, D]
            x_combined = torch.cat([cls_only, masked_patches], dim=1)  # [B, N+1, D]
            
            # === Pass through ViT blocks ===
            for blk in self.vit.blocks:
                x_combined = blk(x_combined)
            
            x_combined = self.vit.norm(x_combined)
            logits = self.vit.head(x_combined[:, 0])  # [B, num_classes]
            
            return logits, patch_masks
        
        else:
            # --- Inference mode: actual truncation ---
            x_out_list = []
            
            for i in range(B):
                # 1) Gather the (0/1) mask for sample i => shape [N]
                sample_mask = patch_masks[i, :, 0]
                selected_indices = (sample_mask > 0).nonzero(as_tuple=False).squeeze(-1)
                
                # 2) Extract patch embeddings (already pos-embedded!) => shape [N, D]
                sample_patches = patch_only[i]  
                selected_patches = sample_patches[selected_indices]
                
                # 3) Combine with CLS token (already pos-embedded!)
                sample_cls = cls_only[i : i+1, :]  # shape [1, D]
                
                # NOTE: We do NOT add positional embeddings again in inference. # CHANGED
                truncated_tokens = torch.cat(
                    [sample_cls, selected_patches],
                    dim=0  # shape [1 + num_selected, D]
                ).unsqueeze(0)  # => [1, (1+num_selected), D]
                
                # 4) Pass through ViT blocks
                for blk in self.vit.blocks:
                    truncated_tokens = blk(truncated_tokens)
                
                truncated_tokens = self.vit.norm(truncated_tokens)
                
                # 5) The head on the CLS token => shape [1, num_classes]
                out_i = self.vit.head(truncated_tokens[:, 0])
                x_out_list.append(out_i)
            
            # Concatenate the outputs => [B, num_classes]
            logits = torch.cat(x_out_list, dim=0)
            return logits

# Same helper for visualizing
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def visualize_masks(image, mask, patch_size=16, save_path=None):
    """Same as before but with save_path"""
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
    ).squeeze().cpu().numpy()
    
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
    # Quick test
    model = SparseViT(num_classes=1000)
    
    # Dummy batch
    dummy_input = torch.randn(2, 3, 224, 224)  # [batch_size, channels, height, width]
    
    # === Test forward pass in eval mode (inference) with actual truncation ===
    model.eval()
    with torch.no_grad():
        try:
            output_inference = model(dummy_input)
            print("✓ Inference (eval) forward pass successful!")
            print("Output shape:", output_inference.shape)  # Should be [2, 1000]
        except Exception as e:
            print("✗ Error during inference forward pass:")
            print(e)
    
    # === Test training mode with soft mask application ===
    model.train()
    try:
        output_train, masks_train = model(dummy_input)
        print("\n✓ Training mode forward pass successful!")
        print("Output shape:", output_train.shape)  # [2, 1000]
        print("Masks shape:", masks_train.shape)     # [2, N, 1]
        print("Average mask value:", masks_train.mean().item())
    except Exception as e:
        print("\n✗ Error during training mode forward pass:")
        print(e)

    # Quick visualization test
    dummy_batch = torch.randn(2, 3, 224, 224)
    N = 196  # 14x14 patches for 224 image with patch_size=16
    test_mask1 = torch.zeros(N, 1)
    test_mask1[::2] = 1  # checkerboard
    
    test_mask2 = torch.zeros(N, 1)
    test_mask2[:N//2] = 1  # top half selected
    
    print("Testing visualization... should see two plots")
    visualize_masks(dummy_batch[0], test_mask1)
    visualize_masks(dummy_batch[1], test_mask2)
    
    import os
    save_dir = "viz_test"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nSaving test visualizations to viz_test/...")
    visualize_masks(dummy_batch[0], test_mask1, 
                   save_path=os.path.join(save_dir, "test_checkerboard.png"))
    visualize_masks(dummy_batch[1], test_mask2,
                   save_path=os.path.join(save_dir, "test_half.png"))
    
    print("Done! Check the viz_test directory for saved images")
