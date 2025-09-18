import timm
import torch.nn as nn

def rename_norm_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Rename norm.* to fc_norm.*
        if k.startswith("norm."):
            new_k = k.replace("norm.", "fc_norm.")
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def get_backbone(model_name='vit_base_patch16_224', pretrained=True, output_dim=64):
    """
    Create a Vision Transformer backbone with optional projection to fixed-dimensional output.

    Args:
        model_name (str): Backbone model name from timm.
        pretrained (bool): Whether to load pretrained weights.
        output_dim (int): Output feature dimension after projection layer.

    Returns:
        nn.Module: Feature extractor + projection layer.

    Some model suggestions:
        'vit_base_patch16_224'
        'vit_small_patch16_224'
        'vit_large_patch14_clip_336.openai' ← CLIP-like ViT
        'swin_base_patch4_window7_224' ← supports varying resolutions well
        'vit_base_patch16_384' ← allows higher input sizes

    """

    # Load ViT-based backbone with no classification head
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,        # Removes classifier head
        global_pool='avg'     # Global average pooling to fixed-size vector
    )

    # Create a sequential model: backbone + optional 64-D projection layer
    model = nn.Sequential(
        backbone,
        nn.LayerNorm(backbone.num_features),    # Normalize output features
        nn.Linear(backbone.num_features, output_dim)  # Project to 64-D
    )

    return model
