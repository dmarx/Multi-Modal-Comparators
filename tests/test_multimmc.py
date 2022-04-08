from mmc.multimmc import MultiMMC
from mmc.modalities import TEXT, IMAGE

def test_init_perceptor():
    perceptor = MultiMMC(TEXT, IMAGE)

def test_load_clip():
    perceptor = MultiMMC(TEXT, IMAGE)
    perceptor.load_model(
        architecture='clip', 
        publisher='openai', 
        id='RN50',
        )

