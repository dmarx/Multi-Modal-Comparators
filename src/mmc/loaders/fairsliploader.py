
"""
Loaders for pretrained CLOOB model by crowsonkb
https://github.com/crowsonkb/cloob-training
"""
from collections import OrderedDict
from pathlib import Path
from platform import architecture
from typing import TYPE_CHECKING

from loguru import logger
import napm
import torch
from torch import hub
from torchvision import transforms

from .basemmcloader import BaseMmcLoader
from ..modalities import TEXT, IMAGE
from ..multimodalcomparator import MultiModalComparator
from ..registry import register_model


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if TYPE_CHECKING:
    import PIL

# The following models are pre-trained on YFCC15M and evaluated on ImageNet-1K (ILSVRC2012).

# ViT-Small (MoCo v3 version w/ 12 vs. 6 heads)
# https://dl.fbaipublicfiles.com/slip/clip_small_25ep.pt
# https://dl.fbaipublicfiles.com/slip/simclr_small_25ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_small_25ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_small_50ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_small_100ep.pt

# ViT-Base - SLIP_VITB16
# https://dl.fbaipublicfiles.com/slip/clip_base_25ep.pt
# https://dl.fbaipublicfiles.com/slip/simclr_base_25ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_base_25ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_base_50ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt

# ViT-Large - SLIP_VITL16
# https://dl.fbaipublicfiles.com/slip/clip_large_25ep.pt
# https://dl.fbaipublicfiles.com/slip/simclr_large_25ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_large_25ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_large_50ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_large_100ep.pt

#url_template = "https://dl.fbaipublicfiles.com/slip/{arch}_{size}_{eps}ep.pt"
# let's use "{arch}_{size}_{eps}ep" as the id
 
def parse_id(slip_id: str):
    arch, size, eps = slip_id.split('_')
    return arch, size, eps

def loader_name_from_id(slip_id: str):
    arch, size, eps = parse_id(slip_id)
    name_str = f"{arch.upper()}_VIT{size[0].upper()}16"
    return name_str

def model_factory_from_id(slip_id: str):
    from SLIP.models import (
        SLIP_VITS16,
        SLIP_VITB16, 
        SLIP_VITL16,
    )
    name = loader_name_from_id(slip_id)
    logger.debug(name)
    #model_factory = globals().get(name)
    model_factory = locals().get(name)
    return model_factory

def url_from_id(slip_id: str):
    return f"https://dl.fbaipublicfiles.com/slip/{slip_id}.pt"

def id_from_url(url):
    fname = url.split('/')[-1]
    return fname.split('.')[0]




#######################################################################################################################

# CC3M
# https://dl.fbaipublicfiles.com/slip/clip_base_cc3m_40ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_base_cc3m_40ep.pt

# CC12M
# https://dl.fbaipublicfiles.com/slip/clip_base_cc12m_35ep.pt
# https://dl.fbaipublicfiles.com/slip/slip_base_cc12m_35ep.pt

#######################################################################################################################

# from SLIP.models import SLIP, SLIP_VITS16, SLIP_VITB16, SLIP_VITL16
# from SLIP.models import CLIP, CLIP_VITS16, CLIP_VITB16, CLIP_VITL16
# from SLIP.models import SIMCLR, SIMCLR_VITS16, SIMCLR_VITB16



# new dependencies (maybe)
# - timm



def fetch_weights(url, namespace, device=DEVICE):
    """
    Downloads the weights from the given url and saves them to the given path.
    If weights have already been downloaded, they are loaded from the path.
    
    :param url: The URL of the checkpoint file
    :param namespace: The name of the model
    :param device: The device to load the weights on
    :return: A dictionary of the weights and biases of the model.
    """
    fname = url.split('/')[-1]
    fpath = Path(hub.get_dir()) / namespace / fname
    try:
        ckpt = torch.load(fpath, map_location=device)
    except FileNotFoundError:
        download(url, fpath)
        ckpt = torch.load(fpath, map_location=device)
    return ckpt


def download(url, fpath):
    """
    If the file doesn't exist, download it
    
    :param url: The URL of the file to download
    :param fpath: The path to the file to download
    """
    if not Path(fpath).exists():
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        hub.download_url_to_file(url, fpath)
    if not Path(fpath).exists():
        raise FileNotFoundError(f"Download failed: {url}")

def fix_param_names_old(ckpt):
    """
    Takes a checkpoint dictionary and removes the "module" prefix from the keys in the state_dict
    
    :param ckpt: the checkpoint file
    """
    logger.debug(ckpt.keys())
    logger.debug(ckpt['args'])
    sd = ckpt['state_dict']
    real_sd = {}
    for k, v in sd.items():
        new_key = '.'.join(k.split('.')[1:]) # strips "module" prefix. sure, why not.
        real_sd[new_key] = v
    del ckpt['state_dict']
    ckpt['state_dict'] = real_sd

def fix_param_names(ckpt):
    # via https://github.com/pixray/pixray/blob/master/slip.py#L127-L128
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    ckpt['state_dict'] = state_dict


#######################################################################################################################




class FairSlipLoaderBase(BaseMmcLoader):
    """
    SLIP models via https://github.com/facebookresearch/SLIP
    """
    def __init__(
        self,
        id,
        architecture,
    ):
        self.architecture = architecture
        self.publisher = 'facebookresearch'
        self.id = id
        self.modalities = (TEXT, IMAGE)
    def _napm_install(self):
        logger.debug('using napm to "install" facebookresearch/SLIP')
        url = "https://github.com/facebookresearch/SLIP"
        napm.pseudoinstall_git_repo(url, env_name='mmc', add_install_dir_to_path=True)
        napm.populate_pythonpaths('mmc')
        from SLIP.models import (
            SLIP_VITS16,
            SLIP_VITB16, 
            SLIP_VITL16
            )
        #from SLIP.tokenizer import SimpleTokenizer


val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

class FairSlipLoader_YFCC15M(FairSlipLoaderBase):
    """
    SLIP models via https://github.com/facebookresearch/SLIP
    """
    def __init__(
        self,
        id,
        architecture,
    ):
        super().__init__(id, architecture)
        self.dataset = 'YFCC15M'
    def load(self, device=DEVICE):
        """
        Returns the MMC associated with this loader.
        """
        self._napm_install()

        model_factory = model_factory_from_id(self.id)
        logger.debug(f"model_factory: {model_factory}")
        ckpt_url = url_from_id(self.id)
        ckpt = fetch_weights(
            url=ckpt_url, 
            namespace='fair_slip', 
            device=device,
            )
        d_args = vars(ckpt['args'])
        kwargs = {k:d_args[k] for k in ('ssl_emb_dim', 'ssl_mlp_dim') if k in d_args}
        logger.debug(kwargs)
        fix_param_names(ckpt)
        model = model_factory(**kwargs)
        model.load_state_dict(ckpt['state_dict'], strict=True)

        from SLIP.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()

        def preprocess_image_extended(*args, **kwargs):
            x = val_transform(*args, **kwargs)
            if x.ndim == 3:
                logger.debug("adding batch dimension")
                x = x.unsqueeze(0)
            return x
        #logger.debug(model)
        mmc = MultiModalComparator(name=str(self), device=device)
        mmc.register_modality(modality=TEXT, projector=model.encode_text, preprocessor=tokenizer)
        mmc.register_modality(modality=IMAGE, projector=model.encode_image, preprocessor= preprocess_image_extended)
        mmc._model = model
        return mmc

# To do: register models

# ViT-Small (MoCo v3 version w/ 12 vs. 6 heads)
model_ids = [
    'clip_small_25ep',
    'simclr_small_25ep',
    'slip_small_25ep',
    'slip_small_50ep',
    'slip_small_100ep',
    'clip_base_25ep',
    'simclr_base_25ep',
    'slip_base_25ep',
    'slip_base_50ep',
    'slip_base_100ep',
    'clip_large_25ep',
    'simclr_large_25ep',
    'slip_large_25ep',
    'slip_large_50ep',
    'slip_large_100ep',
]

for mid in model_ids:
    arch, _, _ = mid.split('_')
    register_model(
        FairSlipLoader_YFCC15M(
            id=mid,
            architecture=arch,
        )
    )
