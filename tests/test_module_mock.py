import pytest
from loguru import logger
import mmc
import PIL
import torch


def test_init():
    from mmc.mock.openai import MockOpenaiClipModule
    from mmc.loaders import OpenAiClipLoader

    ldr = OpenAiClipLoader(id='RN50')
    clip = MockOpenaiClipModule(ldr)

def test_available_models():
    from mmc.mock.openai import MockOpenaiClipModule
    from mmc.loaders import OpenAiClipLoader

    ldr = OpenAiClipLoader(id='RN50')
    clip = MockOpenaiClipModule(ldr)
    assert clip.available_models() == str(ldr)

def test_private_mmc():
    from mmc.mock.openai import MockOpenaiClipModule, MockOpenaiClip
    from mmc.loaders import OpenAiClipLoader
    from mmc.multimodalcomparator import MultiModalComparator

    ldr = OpenAiClipLoader(id='RN50')
    clip = MockOpenaiClipModule(ldr)
    #assert isinstance(clip._clip, MockOpenaiClip)
    assert isinstance(clip._model, MultiModalComparator)


def test_tokenize():
    from mmc.mock.openai import MockOpenaiClipModule
    from mmc.loaders import OpenAiClipLoader
    from clip import tokenize

    ldr = OpenAiClipLoader(id='RN50')
    clip = MockOpenaiClipModule(ldr)
    
    test_text = "foo bar baz"
    tokens = clip.tokenize(test_text)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape[1] == tokenize(test_text).shape[1] 

def test_preprocess_image():
    pass

def test_load():
    pass


