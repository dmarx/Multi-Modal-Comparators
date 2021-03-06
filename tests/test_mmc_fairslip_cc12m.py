import pytest
from mmc.loaders import FairSlipLoader_CC12M as loader
import PIL
from loguru import logger
import torch


loader_args = {
    'id': 'slip_base_cc12m_35ep',
    'architecture': 'slip',
}


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
