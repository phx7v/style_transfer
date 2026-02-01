import torch
import numpy as np
from PIL import Image

from style_transfer.inference.postprocessing import tensor_to_pil
from style_transfer.inference.preprocessing import preprocess_image
from style_transfer.model import TransformNet


def test_full_inference_pipeline():
    model = TransformNet()
    img = Image.new('RGB', (128, 128))

    x = preprocess_image(img, torch.device('cpu'))

    with torch.no_grad():
        y = model(x)

    out = tensor_to_pil(y)

    assert isinstance(out, Image.Image)
    assert out.size == (128, 128)
    assert out.mode == 'RGB'

    np_out = np.array(out)
    assert np_out.min() >= 0
    assert np_out.max() <= 255
    assert np_out.dtype == np.uint8
