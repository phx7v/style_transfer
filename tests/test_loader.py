import torch

from style_transfer.tools.model_loader import load_model
from style_transfer.models.transform_net import TransformNet


def test_load_model_sets_eval(tmp_path):
    dummy_weights = tmp_path / 'w.pt'
    model = TransformNet()
    torch.save(model.state_dict(), dummy_weights)

    loaded = load_model(str(dummy_weights), torch.device('cpu'))

    assert isinstance(loaded, torch.nn.Module)
    assert not loaded.training
