import numpy as np
from PIL import Image
import pytest

from style_transfer.inference.preprocessing import preprocess_image_onnx
from style_transfer.inference.postprocessing import numpy_to_pil


def test_numpy_to_pil_output(dummy_image):
    x = preprocess_image_onnx(dummy_image)

    img = numpy_to_pil(x)

    assert isinstance(img, Image.Image)
    assert img.mode == 'RGB'
    assert img.size[0] == x.shape[3]  # w
    assert img.size[1] == x.shape[2]  # h

    np_img = np.array(img)
    assert np_img.min() >= 0
    assert np_img.max() <= 255
    assert np_img.dtype == np.uint8
