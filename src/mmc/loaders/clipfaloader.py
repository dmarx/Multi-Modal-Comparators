
"""
Loaders for pretrained CLIP models published by OpenAI
"""

#import clip # this should probably be isolated somehow
from loguru import logger
import torch

from .basemmcloader import BaseMmcLoader
from ..modalities import TEXT, IMAGE
from ..multimodalcomparator import MultiModalComparator
from ..registry import REGISTRY, register_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ClipFaLoader(BaseMmcLoader):
    """
    CLIP model trained for the Farsi language (Persian)
    https://github.com/sajjjadayobi/CLIPfa
    """
    def __init__(
        self,
        #id,
        device=DEVICE,
    ):
        self.device=device
        self.architecture = 'clip' # should this be a type too?
        self.publisher = 'sajjjadayobi'
        self.id = 'clipfa'
        self.modalities = (TEXT, IMAGE)
    def load(self, device=None):
        """
        Returns the MMC associated with this loader.
        """
        if device is None:
            device = self.device
        #import clip
        #model, preprocess_image = clip.load(self.id, jit=False, device=device)
        #model.eval()
        #model.requires_grad_(False)
        #model.to(device, memory_format=torch.channels_last)
        #tokenizer = clip.tokenize # clip.simple_tokenizer.SimpleTokenizer()
        #def preprocess_image_extended(*args, **kwargs):
        #    x = preprocess_image(*args, **kwargs)
        #    if x.ndim == 3:
        #        logger.debug("adding batch dimension")
        #        x = x.unsqueeze(0)
        #    return x
        from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer, CLIPFeatureExtractor
        # download pre-trained models
        vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
        preprocessor = CLIPFeatureExtractor.from_pretrained('SajjadAyoubi/clip-fa-vision')
        text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
        tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')
        vision_encoder.to(device)
        text_encoder.to(device)
        #text_embedding = text_encoder(**tokenizer(text, return_tensors='pt')).pooler_output
        #image_embedding = vision_encoder(**preprocessor(image, return_tensors='pt')).pooler_output
        mmc = MultiModalComparator(name=str(self), device=device)
        mmc.register_modality(modality=TEXT, projector=text_encoder, preprocessor=tokenizer)
        mmc.register_modality(modality=IMAGE, projector=vision_encoder, preprocessor=preprocessor)
        mmc._model = vision_encoder
        return mmc


register_model(
    ClipFaLoader()
)