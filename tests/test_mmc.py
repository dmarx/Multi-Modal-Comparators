import pytest
from mmc.loaders import OpenAiClipLoader
import PIL
from loguru import logger
import torch

def test_oai_clip_project_text():
    ldr = OpenAiClipLoader(id='RN50')
    oai_clip = ldr.load()
    projection = oai_clip.project_text("foo bar baz")
    #logger.debug(type(projection))
    #assert projection['modality'] == 'text'
    #logger.debug(projection.shape) # [1 1024]
    assert isinstance(projection, torch.Tensor)

def test_oai_clip_project_img():
    ldr = OpenAiClipLoader(id='RN50')
    oai_clip = ldr.load()
    img = PIL.Image.open("./tests/assets/marley_birthday.jpg").resize((250,200))
    projection = oai_clip.project_image(img)
    #logger.debug(type(projection))
    #assert projection['modality'] == 'image'
    #logger.debug(projection.shape) # [1 1024]
    assert isinstance(projection, torch.Tensor)

def test_oai_clip_supported_modalities():
    ldr = OpenAiClipLoader(id='RN50')
    oai_clip = ldr.load()
    assert oai_clip.supports_text
    assert oai_clip.supports_image
    assert not oai_clip.supports_audio
