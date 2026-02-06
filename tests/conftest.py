import pytest
import torch
from pathlib import Path
from PIL import Image

from style_transfer.models.model import TransformNet


@pytest.fixture
def dummy_weights(tmp_path: Path) -> Path:
    path = tmp_path / 'dummy.pt'
    model = TransformNet()
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def dummy_onnx_path(tmp_path: Path) -> Path:
    return tmp_path / 'dummy.onnx'


@pytest.fixture
def dummy_image() -> Image.Image:
    return Image.new('RGB', (128, 128))
