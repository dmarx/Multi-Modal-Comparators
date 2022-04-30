import pytest
from mmc.loaders import KatCloobLoader as loader
import PIL
from loguru import logger
import torch


#loader_args = {'id':'RN50--cc12m'}
loader_args = {}

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
