import pytest

import mmc

def test_oai_mocking_itself():
    from mmc.mock.openai import MockOpenaiClip
    from mmc.loaders import OpenAiClipLoader

    ldr = OpenAiClipLoader(id='RN50')
    oai_clip = ldr.load()
    model = MockOpenaiClip(oai_clip)
    assert model.visual.input_resolution == 224

