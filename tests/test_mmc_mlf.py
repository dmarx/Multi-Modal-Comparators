import pytest
from mmc.loaders import MlfClipLoader as loader
import PIL
from loguru import logger
import torch

loader_args = {'id':'RN50--cc12m'}

def test_project_text():
    ldr = loader(**loader_args)
    perceptor = ldr.load()
    projection = perceptor.project_text("foo bar baz")
    assert isinstance(projection, torch.Tensor)

def test_project_img():
    ldr = loader(**loader_args)
    perceptor = ldr.load()
    img = PIL.Image.open("./tests/assets/marley_birthday.jpg").resize((250,200))
    projection = perceptor.project_image(img)
    assert isinstance(projection, torch.Tensor)

def test_supported_modalities():
    ldr = loader(**loader_args)
    perceptor = ldr.load()
    assert perceptor.supports_text
    assert perceptor.supports_image
    assert not perceptor.supports_audio

class TestMlfVitb16plus:
    loader_args = {'id':'ViT-B-16-plus-240--laion400m_e32'}

    def test_project_text(self):
        ldr = loader(**self.loader_args)
        perceptor = ldr.load()
        projection = perceptor.project_text("foo bar baz")
        assert isinstance(projection, torch.Tensor)
        logger.debug(projection.shape)

    def test_project_img(self):
        ldr = loader(**self.loader_args)
        perceptor = ldr.load()
        img = PIL.Image.open("./tests/assets/marley_birthday.jpg").resize((300,300))
        projection = perceptor.project_image(img)
        assert isinstance(projection, torch.Tensor)
        logger.debug(projection.shape)

    def test_supported_modalities(self):
        ldr = loader(**self.loader_args)
        perceptor = ldr.load()
        assert perceptor.supports_text
        assert perceptor.supports_image
        assert not perceptor.supports_audio
