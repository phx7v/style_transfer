from collections.abc import Callable

import torch
from PIL import Image
from torch.nn import Module


class AnimeGAN:
    def __init__(
        self,
        generator: Module,
        face2paint_fn: Callable[[Module, Image.Image], Image.Image],
    ) -> None:
        self.generator: Module = generator.eval()
        self.face2paint: Callable[[Module, Image.Image], Image.Image] = face2paint_fn

    def __call__(self, img: Image.Image) -> Image.Image:
        with torch.no_grad():
            return self.face2paint(self.generator, img)
