import torch

from style_transfer.model import TransformNet


def test_transform_net_forward_shape():
    model = TransformNet()
    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape
    assert y.device == x.device
    assert y.dtype == x.dtype
