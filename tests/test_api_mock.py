import pytest
from loguru import logger
import mmc
import PIL
import torch


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


class TestMlfVitb16plus:

    loader_args = {'id':'ViT-B-16-plus-240--laion400m_e32'}

    def test_mock_oai(self):
        from mmc.mock.openai import MockOpenaiClip
        from mmc.loaders import MlfClipLoader

        ldr = MlfClipLoader(**self.loader_args)
        mlf_clip = ldr.load()
        model = MockOpenaiClip(mlf_clip)
        assert model.visual.input_resolution == (240, 240)


    def test_project_text(self):
        from mmc.mock.openai import MockOpenaiClip
        from mmc.loaders import MlfClipLoader
        #from clip.simple_tokenizer import SimpleTokenizer
        import clip

        ldr = MlfClipLoader(**self.loader_args)
        mlf_clip = ldr.load()
        model = MockOpenaiClip(mlf_clip)
        tokens = clip.tokenize("foo bar baz").to(model.device)
        projection = model.encode_text(tokens)
        assert isinstance(projection, torch.Tensor)
        logger.debug(projection.shape)


    def test_project_img(self):
        from mmc.mock.openai import MockOpenaiClip
        from mmc.loaders import MlfClipLoader

        ldr = MlfClipLoader(**self.loader_args)
        mlf_clip = ldr.load()
        model = MockOpenaiClip(mlf_clip)
        im_size = model.visual.input_resolution[0]
        logger.debug(im_size)
        img = torch.rand(1,3,im_size, im_size) # batch x channels x height x width
        #img = torch.rand(3,im_size, im_size) # batch x channels x height x width
        logger.debug(img.shape)
        img = img.to(model.device)
        projection = model.encode_image(img)
        assert isinstance(projection, torch.Tensor)
        logger.debug(projection.shape)










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

