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

# what even is my expected behavior here? It passes the test but, I'm not sure how I'd even use this.
# maybe this should throw an error?
"""
def test_multi_oai_mocking_oai_init():
    from mmc.mock.openai import MockOpenaiClip
    from mmc.multimmc import MultiMMC
    from mmc.modalities import TEXT, IMAGE
    
    perceptor = MultiMMC(TEXT, IMAGE)
    models = [
        dict(
            architecture='clip', 
            publisher='openai', 
            id='RN50',
        ),
        dict(
            architecture='clip', 
            publisher='openai', 
            id='ViT-B/32',
    )]
    for m in models:
        perceptor.load_model(**m)
    #dir(perceptor)
"""

