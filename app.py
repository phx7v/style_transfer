from io import BytesIO
from pathlib import Path

import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image

from inference.postprocessing import numpy_to_pil
from inference.preprocessing import preprocess_image_onnx, resize_to_max_pixels
from models.anime import AnimeGAN

MAX_PIXELS = 3_000_000
WEIGHTS_DIR = Path('models/onnx')


@st.cache_resource
def get_session(model_path: str) -> ort.InferenceSession:
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session


def select_model(weights_dir: Path, prefix: str) -> ort.InferenceSession:
    models = {
        p.stem.replace('transformnet_', '').title(): p
        for p in weights_dir.glob('*.onnx')
    }

    selected_model_name = st.selectbox('Choose style', options=sorted(models.keys()), key=f'{prefix}_model')

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

    if st.session_state.selected_model != selected_model_name:
        st.cache_resource.clear()
        st.session_state.selected_model = selected_model_name

    return get_session(models[selected_model_name])


def load_image_from_input(prefix: str) -> Image.Image:
    option = st.radio('Choose image source', ('Upload image', 'Take photo'), key=f'{prefix}_source')

    content_file = (
        st.file_uploader('Upload image', ['png', 'jpg', 'jpeg'], key=f'{prefix}_file')
        if option == 'Upload image'
        else st.camera_input('Take photo', key=f'{prefix}_camera')
    )

    if not content_file:
        return None

    content_image = Image.open(content_file).convert('RGB')

    return content_image


def validate_and_resize_image(content_image: Image.Image, prefix: str) -> Image.Image | None:
    w, h = content_image.size
    if w * h <= MAX_PIXELS:
        return content_image

    st.error(f'Image too large: {w} x {h}. Max allowed: {MAX_PIXELS / 1e6:.1f} MP.')

    if st.button('Resize image', key=f'{prefix}_resize'):
        resized_image = resize_to_max_pixels(content_image, MAX_PIXELS)
        return resized_image


def transfer_net_tab(prefix: str = 'transfer') -> None:
    model = select_model(WEIGHTS_DIR, prefix)

    content_img = load_image_from_input(prefix)
    if content_img is None:
        return

    content_img = validate_and_resize_image(content_img, prefix)
    if content_img is None:
        return

    st.image(content_img, caption='Content Image', width='content')

    content_img = preprocess_image_onnx(content_img)
    with st.spinner('Applying style...'):
            output = model.run(None, {'input': content_img})

    output_img = numpy_to_pil(output[0])

    st.image(output_img, caption='Stylized Image', width='content')

    buf = BytesIO()
    output_img.save(buf, format='PNG')

    st.download_button(
        label='Download result', data=buf.getvalue(), file_name='output.png', mime='image/png',
        key=f'{prefix}_download',
    )


@st.cache_resource
def load_animegan() -> AnimeGAN:
    generator = torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained='face_paint_512_v2')
    face2paint = torch.hub.load('bryandlee/animegan2-pytorch:main', 'face2paint', size=512)
    return AnimeGAN(generator, face2paint)


def anime_tab(prefix: str = 'anime') -> None:
    st.text('Image will be automatically resized by the net to 512 x 512')

    model = load_animegan()

    content_img = load_image_from_input(prefix)
    if content_img is None:
        return

    content_img = validate_and_resize_image(content_img, prefix)
    if content_img is None:
        return

    st.image(content_img, caption='Content Image', width='content')

    with st.spinner('Applying style...'):
        output_img = model(content_img)

    st.image(output_img, caption='Stylized Image', width='content')

    buf = BytesIO()
    output_img.save(buf, format='PNG')

    st.download_button(
        label='Download result', data=buf.getvalue(), file_name='output.png', mime='image/png',
        key=f'{prefix}_download',
    )


def main() -> None:
    st.title('Style Transfer')

    tab = st.radio(
        'Choose mode',
        ['TransferNet Models', 'Anime faces'],
        horizontal=True,
        key='active_tab',
    )

    if tab == 'TransferNet Models':
        transfer_net_tab()
    else:
        anime_tab()


if __name__ == '__main__':
    main()
