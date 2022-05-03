
"""
Loaders for pretrained CLOOB model by crowsonkb
https://github.com/crowsonkb/cloob-training
"""

# importing this first is necessary for cloob to be available
import napm

from loguru import logger
import torch

from .basemmcloader import BaseMmcLoader
from ..modalities import TEXT, IMAGE
from ..multimodalcomparator import MultiModalComparator
from ..registry import REGISTRY, register_model

from torchvision.transforms import ToTensor

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PIL

class KatCloobLoader(BaseMmcLoader):
    """
    CLOOB models by crowsonkb, initially trained on LAION datasets
    https://github.com/crowsonkb/cloob-training
    """
    def __init__(
        self,
        id='cloob_laion_400m_vit_b_16_32_epochs',
    ):
        self.architecture = 'cloob' # should this be a type too?
        self.publisher = 'crowsonkb'
        self.id = id
        self.modalities = (TEXT, IMAGE)
    def load(self, device=DEVICE):
        """
        Returns the MMC associated with this loader.
        """
        logger.debug('using napm to "install" katCLOOB')
        url = "https://github.com/crowsonkb/cloob-training"
        napm.pseudoinstall_git_repo(url, env_name='mmc', package_name='cloob')
        napm.populate_pythonpaths('mmc')
        from cloob.cloob_training import model_pt, pretrained
        
        config = pretrained.get_config(self.id)
        model = model_pt.get_pt_model(config)
        checkpoint = pretrained.download_checkpoint(config)
        model.load_state_dict(model_pt.get_pt_params(config, checkpoint))
        model.eval().requires_grad_(False).to(device)
        d_im = config['image_encoder']['image_size']

        def _preprocess_closure(img: "PIL.Image.Image") -> torch.Tensor:
            img = img.resize((d_im, d_im)).convert('RGB')
            t_img = ToTensor()(img)
            if t_img.ndim == 3:
                t_img = t_img.unsqueeze(0)
            t_img = t_img.to(device)
            return model.normalize(t_img)
        
        mmc = MultiModalComparator(name=str(self), device=device)
        mmc.register_modality(modality=TEXT, projector=model.text_encoder, preprocessor=model.tokenize)
        mmc.register_modality(modality=IMAGE, projector=model.image_encoder, preprocessor=_preprocess_closure)
        mmc._model = model
        return mmc

try:
    from cloob.cloob_training import model_pt, pretrained
    for model_name in pretrained.list_configs():
        register_model(
            KatCloobLoader(id=model_name)
        )
except:
    logger.warning(
        "unable to import cloob: bypassing loader registration. You can still isntall and load cloob via:"
        "`model = KatCloobLoader(id=...).load()`"
    )