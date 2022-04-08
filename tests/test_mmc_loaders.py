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
    oai_clip = OpenAiClipLoader(id='RN50')