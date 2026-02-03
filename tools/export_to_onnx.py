from collections.abc import Callable

import torch


def export_to_onnx(
    model_loader: Callable[[str, torch.device], torch.nn.Module],
    weights_path: str,
    onnx_path: str,
) -> None:
    device = torch.device('cpu')
    model = model_loader(weights_path, device)
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=15,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'H', 3: 'W'}, 'output': {2: 'H', 3: 'W'}},
    )
