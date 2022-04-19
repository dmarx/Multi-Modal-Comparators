import pytest
from mmc.loaders import SBertClipLoader as loader
import PIL
from loguru import logger
import torch

# for some reason SBERT is returning np.ndarrays instead of tensors.
# might be sentence-transformer wonkiness that could be resolved by
# using hugginface/transformers directly.

#loader_args = {'id':'ViT-B-32-multilingual-v1'}
loader_args = {}

def test_project_text():
    ldr = loader(**loader_args)
    perceptor = ldr.load()
    projection = perceptor.project_text("foo bar baz")
    print(type(projection))
    assert isinstance(projection, torch.Tensor)

def test_project_img():
    ldr = loader(**loader_args)
    perceptor = ldr.load()
    img = PIL.Image.open("./tests/assets/marley_birthday.jpg").resize((250,200))
    projection = perceptor.project_image(img)
    print(type(projection))
    assert isinstance(projection, torch.Tensor)

def test_supported_modalities():
    ldr = loader(**loader_args)
    perceptor = ldr.load()
    assert perceptor.supports_text
    assert perceptor.supports_image
    assert not perceptor.supports_audio
