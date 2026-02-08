import torch

from models.transform_net import TransformNet


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = TransformNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
