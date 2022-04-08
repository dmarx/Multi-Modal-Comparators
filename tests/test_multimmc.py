from mmc.multimmc import MultiMMC
from mmc.modalities import TEXT, IMAGE, AUDIO

def test_init_perceptor():
    perceptor = MultiMMC(TEXT, IMAGE)

def test_load_clip():
    perceptor = MultiMMC(TEXT, IMAGE)
    perceptor.load_model(
        architecture='clip', 
        publisher='openai', 
        id='RN50',
        )

def test_supports_modality_property():
    perceptor = MultiMMC(TEXT, IMAGE)
    perceptor.load_model(
        architecture='clip', 
        publisher='openai', 
        id='RN50',
        )
    assert perceptor.supports_image
    assert perceptor.supports_text
    assert not perceptor.supports_audio

def test_supports_modality_function():
    perceptor = MultiMMC(TEXT, IMAGE)
    perceptor.load_model(
        architecture='clip', 
        publisher='openai', 
        id='RN50',
        )
    assert perceptor.supports_modality(IMAGE)
    assert perceptor.supports_modality(TEXT)
    assert not perceptor.supports_modality(AUDIO)


def test_supports_modality_name():
    perceptor = MultiMMC(TEXT, IMAGE)
    perceptor.load_model(
        architecture='clip', 
        publisher='openai', 
        id='RN50',
        )
    assert perceptor._supports_mode('image')
    assert perceptor._supports_mode('text')
    assert not perceptor._supports_mode('audio')
