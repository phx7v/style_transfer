import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = (tensor + 1) / 2
    img = tensor.squeeze().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)
