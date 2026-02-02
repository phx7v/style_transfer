from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def pad_to_size_multiple_of_eight(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[2], x.shape[3]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x


def get_coorect_size(img: Image.Image, max_size: int = 512) -> Tuple[int, int]:
    w, h = img.size
    new_h, new_w = h, w
    if max(w, h) > max_size:
        if w >= h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)

    return new_h, new_w


def preprocess_image(img: Image.Image, device: torch.device) -> torch.Tensor:
    size = get_coorect_size(img)
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
        ]
    )
    x = transform(img.convert('RGB')).unsqueeze(0).to(device)

    return pad_to_size_multiple_of_eight(x)
