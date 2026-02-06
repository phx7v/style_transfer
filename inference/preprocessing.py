import numpy as np
from PIL import Image


def preprocess_image_onnx(img: Image.Image) -> np.ndarray:
    # Convert to numpy float32 [0,1]
    x = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]

    # HWC -> CHW
    x = x.transpose(2, 0, 1)  # [3, H, W]

    # add batch dimension
    x = x[np.newaxis, :, :, :]  # [1, 3, H, W]

    x = x * 2 - 1

    _, _, h, w = x.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    return x


def resize_to_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    pixels = w * h

    if pixels <= max_pixels:
        return img

    scale = (max_pixels / pixels) ** 0.5

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return img.resize((new_w, new_h), Image.BICUBIC)
