import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def preprocess_image(img: Image.Image, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
        ]
    )
    x = transform(img.convert('RGB')).unsqueeze(0).to(device)

    # padding to a size multiple of 8
    h, w = x.shape[2], x.shape[3]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x
