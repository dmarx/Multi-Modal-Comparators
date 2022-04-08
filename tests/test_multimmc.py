from mmc.multimmc import MultiMMC
from mmc.modalities import TEXT, IMAGE, AUDIO
import PIL

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

def test_compare_text2img():
    perceptor = MultiMMC(TEXT, IMAGE)
    perceptor.load_model(
        architecture='clip', 
        publisher='openai', 
        id='RN50',
        )
    text_pos = "a photo of a dog"
    text_neg = "a painting of a cat"
    img = PIL.Image.open('./tests/assets/marley_birthday.jpg').resize((250,200))
    v_pos = perceptor.compare(image=img, text=text_pos)
    v_neg = perceptor.compare(image=img, text=text_pos)
    logger.debug(v_pos, v_neg)
    assert v_pos > v_neg
