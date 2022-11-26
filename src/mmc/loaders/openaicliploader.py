
"""
Loaders for pretrained CLIP models published by OpenAI
"""

import clip # this should probably be isolated somehow
from loguru import logger
import torch

from .basemmcloader import BaseMmcLoader
from ..modalities import TEXT, IMAGE
from ..multimodalcomparator import MultiModalComparator
from ..registry import REGISTRY, register_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class OpenAiClipLoader(BaseMmcLoader):
    """
    Generic class for loading CLIP models published by OpenAI.
    There should be a one-to-one mapping between loader objects
    and specific sets of pretrained weights (distinguished by the "id" field)
    """
    def __init__(
        self,
        id,
        device=DEVICE,
    ):
        self.architecture = 'clip' # should this be a type too?
        self.publisher = 'openai'
        self.id = id
        self.modalities = (TEXT, IMAGE)
        self.device = device

    def load(self, device=None):
        """
        Returns the MMC associated with this loader.
        """
        if device is None:
            device = self.device
        import clip
        model, preprocess_image = clip.load(self.id, jit=False, device=device)
        model.eval()
        model.requires_grad_(False)
        #model.to(device, memory_format=torch.channels_last)
        tokenizer = clip.tokenize # clip.simple_tokenizer.SimpleTokenizer()
        def preprocess_image_extended(*args, **kwargs):
            x = preprocess_image(*args, **kwargs)
            if x.ndim == 3:
                logger.debug("adding batch dimension")
                x = x.unsqueeze(0)
            return x
        mmc = MultiModalComparator(name=str(self), device=device)
        mmc.register_modality(modality=TEXT, projector=model.encode_text, preprocessor=tokenizer)
        mmc.register_modality(modality=IMAGE, projector=model.encode_image, preprocessor=preprocess_image_extended)
        mmc._model = model
        return mmc


for model_name in clip.available_models():
    #REGISTRY.loaders.append(
    #    OpenAiClipLoader(id=model_name)
    #)
    register_model(
        OpenAiClipLoader(id=model_name)
    )