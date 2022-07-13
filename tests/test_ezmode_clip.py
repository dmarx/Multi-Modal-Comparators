import pytest

def test_import():
    from mmc.ez.CLIP import clip

def test_available_models():
    from mmc.ez.CLIP import clip
    clip.available_models()

def test_load_openai_ez():
    from mmc.ez.CLIP import clip
    model, preprocessor = clip.load('RN50')
    assert model
    assert preprocessor

def test_load_openai_alias():
    from mmc.ez.CLIP import clip
    model, preprocessor = clip.load('[clip - openai - RN50]')
    assert model
    assert preprocessor
