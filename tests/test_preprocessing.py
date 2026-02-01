import torch
from PIL import Image

from style_transfer.inference.preprocessing import preprocess_image


def test_preprocess_image_shape_and_range():
    img = Image.new('RGB', (257, 259))
    device = torch.device('cpu')

    x = preprocess_image(img, device)

    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 1
    assert x.shape[1] == 3
    assert x.shape[2] % 8 == 0
    assert x.shape[3] % 8 == 0
    assert x.min() >= -1
    assert x.max() <= 1
