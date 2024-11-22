'''
Author: ssp
Date: 2024-10-29 23:18:42
LastEditTime: 2024-11-20 20:29:34
'''
import torch

from .modules.lpips import LPIPS
def lpips(x: torch.Tensor,
          y: torch.Tensor,
          mask: torch.Tensor = None,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS) with optional masking.

    Arguments:
        x, y (torch.Tensor): the input tensors to compare. Shape: (B, C, H, W).
        mask (torch.Tensor): optional mask tensor. Shape: (B, 1, H, W) or (B, H, W). Values: 0 for masked, 1 for valid pixels.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.

    Returns:
        torch.Tensor: the masked LPIPS score.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    
    # Ensure the mask is in the correct format
    if mask is not None:
        if mask.dim() == 3:  # If mask has shape (B, H, W), add channel dimension
            mask = mask.unsqueeze(1)
        mask = mask.to(device).float()  # Ensure mask is on the same device and type
    
    # Compute raw LPIPS score
    raw_lpips = criterion(x, y)  # Output shape: (B,)
    
    if mask is not None:
        # Apply mask to inputs
        masked_x = x * mask
        masked_y = y * mask
        
        # Recompute LPIPS score on masked inputs
        masked_lpips = criterion(masked_x, masked_y)
        
        # Normalize by the number of valid pixels to avoid scale issues
        valid_pixel_count = mask.sum(dim=(1, 2, 3))  # Count non-masked pixels per batch
        normalized_lpips = masked_lpips / (valid_pixel_count + 1e-8)  # Avoid division by zero
        
        return normalized_lpips
    else:
        return raw_lpips


# def lpips(x: torch.Tensor,
#           y: torch.Tensor,
#           net_type: str = 'alex',
#           version: str = '0.1'):
#     r"""Function that measures
#     Learned Perceptual Image Patch Similarity (LPIPS).

#     Arguments:
#         x, y (torch.Tensor): the input tensors to compare.
#         net_type (str): the network type to compare the features: 
#                         'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
#         version (str): the version of LPIPS. Default: 0.1.
#     """
#     device = x.device
#     criterion = LPIPS(net_type, version).to(device)
#     return criterion(x, y)
