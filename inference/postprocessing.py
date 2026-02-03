import numpy as np
from PIL import Image


def numpy_to_pil(x: np.ndarray) -> Image.Image:
    # remove batch dimension
    x = x[0]  # [3, H, W]

    # CHW -> HWC
    x = x.transpose(1, 2, 0)  # [H, W, 3]

    # [-1, 1] -> [0, 1]
    x = (x + 1) / 2

    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    return Image.fromarray(x)
