import pytest

import mmc

def test_oai_mocking_itself():
    from mmc.mock.openai import MockOpenaiClip
    from mmc.loaders import OpenAiClipLoader

    ldr = OpenAiClipLoader(id='RN50')
    oai_clip = ldr.load()
    model = MockOpenaiClip(oai_clip)
    assert model.visual.input_resolution == 224

def test_mlf_mocking_oai():
    from mmc.mock.openai import MockOpenaiClip
    from mmc.loaders import MlfClipLoader

    ldr = MlfClipLoader(id='RN50--yfcc15m')
    mlf_clip = ldr.load()
    model = MockOpenaiClip(mlf_clip)
    assert model.visual.input_resolution == 224
