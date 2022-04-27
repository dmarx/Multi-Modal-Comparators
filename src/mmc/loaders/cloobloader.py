
"""
Loaders for pretrained CLOOB model by crowsonkb
https://github.com/crowsonkb/cloob-training
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
    CLOOB models by crowsonkb, initially trained on LAION datasets
    https://github.com/crowsonkb/cloob-training
    """
    def __init__(
        self,
        id,
    ):
        self.architecture = 'cloob' # should this be a type too?
        self.publisher = 'crowsonkb'
        self.id = id
        self.modalities = (TEXT, IMAGE)
    def load(self, device=DEVICE):
        """
        Returns the MMC associated with this loader.
        """
        from cloob_training import model_pt, pretrained
        
        config = pretrained.get_config(self.id)
        model = model_pt.get_pt_model(config)
        checkpoint = pretrained.download_checkpoint(config)
        model.load_state_dict(model_pt.get_pt_params(config, checkpoint))
        model.eval().requires_grad_(False).to(device)
        
        mmc = MultiModalComparator(name=str(self), device=device)
        mmc.register_modality(modality=TEXT, projector=model.text_encoder, preprocessor=model.tokenize)
        mmc.register_modality(modality=IMAGE, projector=model.image_encoder, preprocessor=model.normalize)
        mmc._model = model
        return mmc


register_model(
    ClipFaLoader()
)