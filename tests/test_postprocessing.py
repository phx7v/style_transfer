import torch
import numpy as np
from PIL import Image

from style_transfer.inference.postprocessing import tensor_to_pil


def test_tensor_to_pil_output():
    tensor = torch.rand(1, 3, 64, 64) * 2 - 1

    img = tensor_to_pil(tensor)

    assert isinstance(img, Image.Image)
    assert img.mode == 'RGB'
    assert img.size == (64, 64)

    np_img = np.array(img)
    assert np_img.min() >= 0
    assert np_img.max() <= 255
