import numpy as np
import pytest
from PIL import Image

from style_transfer.inference.preprocessing import preprocess_image_onnx, resize_to_max_pixels


def test_preprocess_image_onnx_shape_and_range(dummy_image):
    x = preprocess_image_onnx(dummy_image)

    assert isinstance(x, np.ndarray)
    assert x.shape[0] == 1  # batch
    assert x.shape[1] == 3  # channels
    assert x.shape[2] % 8 == 0  # H padded to multiple of 8
    assert x.shape[3] % 8 == 0  # W padded to multiple of 8

    assert x.min() >= -1
    assert x.max() <= 1


def test_resize_noop_when_under_limit():
    img = Image.new('RGB', (500, 500))  # 250_000 pixels
    max_pixels = 1_000_000

    out = resize_to_max_pixels(img, max_pixels)

    assert out is img
    assert out.size == (500, 500)


def test_resize_down_when_over_limit():
    img = Image.new('RGB', (4000, 3000))  # 12 MP
    max_pixels = 1_000_000

    out = resize_to_max_pixels(img, max_pixels)

    w, h = out.size
    assert w * h <= max_pixels
    assert w < 4000
    assert h < 3000


def test_resize_preserves_aspect_ratio():
    img = Image.new('RGB', (6000, 4000))
    max_pixels = 1_000_000

    out = resize_to_max_pixels(img, max_pixels)

    orig_ratio = img.size[0] / img.size[1]
    new_ratio = out.size[0] / out.size[1]

    assert orig_ratio == pytest.approx(new_ratio, abs=1e-2)


def test_resize_never_returns_zero_dimension():
    img = Image.new('RGB', (10_000, 1))
    max_pixels = 10

    out = resize_to_max_pixels(img, max_pixels)

    w, h = out.size
    assert w >= 1
    assert h >= 1


def test_resize_exactly_at_limit():
    img = Image.new('RGB', (1000, 1000))  # 1 MP
    max_pixels = 1_000_000

    out = resize_to_max_pixels(img, max_pixels)

    assert out is img
    assert out.size == (1000, 1000)
