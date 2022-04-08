import pytest

def test_loader_import():
    from mmc.loaders import OpenAiClipLoader

def test_load_oai_clip():
    from mmc.loaders import OpenAiClipLoader
    oai_clip = OpenAiClipLoader(id='RN50')