from io import BytesIO

import streamlit as st
import torch
from PIL import Image

from inference.loader import load_model
from inference.postprocessing import tensor_to_pil
from inference.preprocessing import preprocess_image


@st.cache_resource
def get_model() -> tuple[torch.nn.Module, torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('weights/vg7.pt', device)
    return model, device


def main() -> None:
    st.title('Style Transfer: Van Gogh')

    model, device = get_model()

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

    content = preprocess_image(content_img, device)

    with torch.no_grad():
        output = model(content)

    output_img = tensor_to_pil(output)
    st.image(output_img, caption='Stylized Image', width='content')

    buf = BytesIO()
    output_img.save(buf, format='PNG')

    st.download_button(label='Download result', data=buf.getvalue(), file_name='output.png', mime='image/png')


if __name__ == '__main__':
    main()
