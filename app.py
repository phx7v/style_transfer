from io import BytesIO
from pathlib import Path

import onnxruntime as ort
import streamlit as st
from PIL import Image

from inference.postprocessing import numpy_to_pil
from inference.preprocessing import preprocess_image_onnx


@st.cache_resource
def get_session(model_path: str) -> ort.InferenceSession:
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session


def main() -> None:
    st.title('Style Transfer')

    weights_dir = Path('models/onnx')

    models = {p.stem.replace('transformnet_', '').title(): p for p in weights_dir.glob('*.onnx')}

    selected_model_name = st.selectbox('Choose style', options=sorted(models.keys()))

    model = get_session(models[selected_model_name])

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
