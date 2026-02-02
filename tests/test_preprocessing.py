import torch
from PIL import Image
from torchvision import transforms

from style_transfer.inference.preprocessing import preprocess_image, pad_to_size_multiple_of_eight, get_coorect_size


def test_get_coorect_size():
    img = Image.new('RGB', (843, 954))
    size = get_coorect_size(img)
    assert size == (512, 452)


def test_pad_to_size_multiple_of_eight():
    img = Image.new('RGB', (257, 259))
    device = torch.device('cpu')

    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img.convert('RGB')).unsqueeze(0).to(device)

    assert isinstance(x, torch.Tensor)

    x = pad_to_size_multiple_of_eight(x)

    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 1
    assert x.shape[1] == 3
    assert x.shape[2] % 8 == 0
    assert x.shape[3] % 8 == 0


def test_preprocess_image_shape_and_range():
    img = Image.new('RGB', (257, 259))
    device = torch.device('cpu')

    x = preprocess_image(img, device)

    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 1
    assert x.shape[1] == 3
    assert x.shape[2] % 8 == 0
    assert x.shape[3] % 8 == 0
    assert x.min() >= -1
    assert x.max() <= 1


def test_preprocess_image_too_big():
    h, w = 1024, 512
    img = Image.new('RGB', (w, h))
    device = torch.device('cpu')

    x = preprocess_image(img, device)

    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 1
    assert x.shape[1] == 3
    assert x.shape[2] == h / 2
    assert x.shape[3] == w / 2
    assert x.min() >= -1
    assert x.max() <= 1
