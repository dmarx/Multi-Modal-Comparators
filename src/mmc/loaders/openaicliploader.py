
"""
Loaders for pretrained CLIP models published by OpenAI
"""

import clip # this should probably be isolated somehow
import torch

from .basemmcloader import BaseMmcLoader
from ..modalities import TEXT, IMAGE
from ..multimodalcomparator import MultiModalComparator
from ..registry import REGISTRY


class OpenAiClipLoader(BaseMmcLoader):
    """
    Generic class for loading CLIP models published by OpenAI.
    There should be a one-to-one mapping between loader objects
    and specific sets of pretrained weights (distinguished by the "id" field)
    """
    def __init__(
        self,
        id,
    ):
        self.architecture = 'clip' # should this be a type too?
        self.publisher = 'openai'
        self.id = id
        self.modalities = (TEXT, IMAGE)
    def load(self, device=DEVICE):
        """
        Returns the MMC associated with this loader.
        """
        import clip
        model, preprocess_image = clip.load(self.id, jit=False)
        model.requires_grad_(False)
        model.to(device, memory_format=torch.channels_last)
        tokenizer = clip.tokenize # clip.simple_tokenizer.SimpleTokenizer()
        mmc = MultiModalComparator(name=str(self), device=device)
        mmc.register_modality(modality=TEXT, projector=model.encode_text, preprocessor=tokenizer)
        mmc.register_modality(modality=IMAGE, projector=model.encode_image, preprocessor=preprocess_image)
        return mmc


for model_name in clip.available_models():
  REGISTRY.loaders.append(
      OpenAiClipLoader(id=model_name)
      )