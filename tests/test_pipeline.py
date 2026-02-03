import torch
import onnx
import onnxruntime as ort
import numpy as np
from style_transfer.models.model import TransformNet
from style_transfer.inference.postprocessing import numpy_to_pil
from style_transfer.inference.preprocessing import preprocess_image_onnx


def test_full_pipeline(dummy_weights, dummy_onnx_path, dummy_image):
    # PyTorch -> ONNX
    model = TransformNet()
    state = torch.load(dummy_weights)
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model,
        dummy_input,
        str(dummy_onnx_path),
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'H', 3: 'W'}, 'output': {2: 'H', 3: 'W'}},
    )

    onnx_model = onnx.load(str(dummy_onnx_path))
    onnx.checker.check_model(onnx_model)

    x = preprocess_image_onnx(dummy_image)

    # ONNX inference
    sess = ort.InferenceSession(str(dummy_onnx_path))
    outputs = sess.run(None, {'input': x})
    y = outputs[0]

    out_img = numpy_to_pil(y)

    assert out_img.size == (128, 128)
    assert out_img.mode == 'RGB'
    np_out = np.array(out_img)
    assert np_out.min() >= 0
    assert np_out.max() <= 255
    assert np_out.dtype == np.uint8
