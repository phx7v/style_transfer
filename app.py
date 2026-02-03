from io import BytesIO

import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image

from inference.postprocessing import numpy_to_pil
from inference.preprocessing import preprocess_image_onnx


@st.cache_resource
def get_model(model_path: str) -> torch.nn.Module:
    model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return model


def main() -> None:
    st.title('Style Transfer: Van Gogh')

    model = get_model('models/onnx/transformnet_vg.onnx')

    option = st.radio('Choose image source', ('Upload image', 'Take photo'))

    content_file = (
        st.file_uploader('Upload image', ['png', 'jpg', 'jpeg'])
        if option == 'Upload image'
        else st.camera_input('Take photo')
    )

    if not content_file:
        return

    content_img = Image.open(content_file).convert('RGB')
    st.image(content_img, caption='Content Image', width='content')

    content = preprocess_image_onnx(content_img)

    output = model.run(None, {'input': content})

    output_img = numpy_to_pil(output[0])
    st.image(output_img, caption='Stylized Image', width='content')

    buf = BytesIO()
    output_img.save(buf, format='PNG')

    st.download_button(label='Download result', data=buf.getvalue(), file_name='output.png', mime='image/png')


if __name__ == '__main__':
    main()
