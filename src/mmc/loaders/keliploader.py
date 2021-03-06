
"""
Loaders for pretrained Korean CLIP (KELIP) published by navervision
https://github.com/navervision/KELIP
"""

#import clip # this should probably be isolated somehow
from loguru import logger
import torch

from .basemmcloader import BaseMmcLoader
from ..modalities import TEXT, IMAGE
from ..multimodalcomparator import MultiModalComparator
from ..registry import REGISTRY, register_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#class ClipFaLoader(BaseMmcLoader):
class ClipKelipLoader(BaseMmcLoader):
    """
    CLIP model trained for Korean and English languages
    https://github.com/navervision/KELIP
    """
    def __init__(
        self,
        id='kelip_ViT-B/32',
    ):
        self.architecture = 'clip' # should this be a type too?
        self.publisher = 'navervision'
        self.id = id
        self.modalities = (TEXT, IMAGE)
    def load(self, device=DEVICE):
        """
        Returns the MMC associated with this loader.
        """
        import kelip 
        _id = self.id.replace('kelip_','')
        model, preprocess_img, tokenizer = kelip.build_model(_id)
        
        mmc = MultiModalComparator(name=str(self), device=device)
        mmc.register_modality(modality=TEXT, projector=model.encode_text, preprocessor=tokenizer)
        mmc.register_modality(modality=IMAGE, projector=model.encode_image, preprocessor=preprocess_img)
        mmc._model = model
        return mmc


register_model(
    #They don't have a systematic way for listing their weights it for now and only support ViT-B/32 
    ClipKelipLoader(id='kelip_ViT-B/32')
)