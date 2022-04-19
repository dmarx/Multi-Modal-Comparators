import pytest

from mmc.modalities import TEXT, IMAGE


def test_loader_import():
    from mmc.loaders import OpenAiClipLoader


def test_loader_attrs():
    from mmc.loaders import OpenAiClipLoader
    ldr = OpenAiClipLoader(id='RN50')
    assert ldr.architecture == 'clip'
    assert ldr.publisher == 'openai'
    assert ldr.id =='RN50'
    assert len(ldr.modalities) == 2
    assert ldr.supports_modality(TEXT)
    assert ldr.supports_modality(IMAGE)


def test_load_oai_clip():
    from mmc.loaders import OpenAiClipLoader
    ldr = OpenAiClipLoader(id='RN50')
    oai_clip = ldr.load()

def test_load_mlf_clip():
    from mmc.loaders import MlfClipLoader
    ldr = MlfClipLoader(id='RN50--cc12m')
    mlf_clip = ldr.load()

## Models below pass load but fail inference tests.
# Commonality here I think is models loaded from huggingface

def test_load_sbert_mclip():
    from mmc.loaders import SBertClipLoader
    ldr = SBertClipLoader()
    sbert_mclip = ldr.load()

def test_load_clipfa():
    from mmc.loaders import ClipFaLoader
    ldr = ClipFaLoader()
    farsi_clip = ldr.load()