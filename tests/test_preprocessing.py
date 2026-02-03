import numpy as np
from style_transfer.inference.preprocessing import preprocess_image_onnx


def test_preprocess_image_onnx_shape_and_range(dummy_image):
    x = preprocess_image_onnx(dummy_image)

    assert isinstance(x, np.ndarray)
    assert x.shape[0] == 1  # batch
    assert x.shape[1] == 3  # channels
    assert x.shape[2] % 8 == 0  # H padded to multiple of 8
    assert x.shape[3] % 8 == 0  # W padded to multiple of 8

    assert x.min() >= -1
    assert x.max() <= 1

