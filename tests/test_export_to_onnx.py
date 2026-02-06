import numpy as np
import onnx
import onnxruntime as ort
import pytest

from tools.export_to_onnx import export_to_onnx
from tools.model_loader import load_model


def test_export_creates_file(dummy_onnx_path, dummy_weights):
    export_to_onnx(load_model, dummy_weights, str(dummy_onnx_path))
    assert dummy_onnx_path.exists()

def test_onnx_is_valid(dummy_onnx_path, dummy_weights):
    export_to_onnx(load_model, dummy_weights, str(dummy_onnx_path))
    model = onnx.load(str(dummy_onnx_path))
    onnx.checker.check_model(model)

def test_onnx_inference(dummy_onnx_path, dummy_weights):
    export_to_onnx(load_model, dummy_weights, str(dummy_onnx_path))
    sess = ort.InferenceSession(str(dummy_onnx_path))

    dummy_input = np.random.randn(1, 3, 512, 512).astype(np.float32)
    outputs = sess.run(None, {'input': dummy_input})

    out = outputs[0]
    assert out.shape[0] == 1
    assert out.shape[1] == 3
    assert out.shape[2] % 8 == 0
    assert out.shape[3] % 8 == 0
