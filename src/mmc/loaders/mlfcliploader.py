"""
Loaders for pretrained CLIP models published by MLFoundations
https://github.com/mlfoundations/open_clip
"""


#import clip # this should probably be isolated somehow
import open_clip
from loguru import logger
import torch

from .basemmcloader import BaseMmcLoader
from ..modalities import TEXT, IMAGE
from ..multimodalcomparator import MultiModalComparator
from ..registry import register_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MlfClipLoader(BaseMmcLoader):
    """
    Generic class for loading CLIP models published by MLFoundations.
    https://github.com/mlfoundations/open_clip

    There should be a one-to-one mapping between loader objects
    and specific sets of pretrained weights (distinguished by the "id" field).
    """
    def __init__(
        self,
        id,
        metadata=None,
    ):
        self.architecture = 'clip' # should this be a type too?
        self.publisher = 'mlfoundations'
        self.id = id
        self.modalities = (TEXT, IMAGE)
        self.metadata = {} if metadata is None else metadata

    def load(self, device=DEVICE):
        """
        Returns the MMC associated with this loader.
        """
        import open_clip
        #model, preprocess_image = clip.load(self.id, jit=False, device=device)
        model_name, dataset = self.id.split('--')
        #model, _, preprocess_image = open_clip.create_model_and_transforms(
        model, preprocess_image, _ = open_clip.create_model_and_transforms(
            model_name=model_name, 
            pretrained=dataset)

        model.requires_grad_(False)
        model.eval()
        model.to(device, memory_format=torch.channels_last)
        #tokenizer = clip.tokenize # clip.simple_tokenizer.SimpleTokenizer()
        tokenizer = open_clip.tokenize # clip.simple_tokenizer.SimpleTokenizer()
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


for model_name, dataset in open_clip.list_pretrained():
    metadata = {}
    if model_name == "ViT-B-16-plus-240":
        metadata['input_resolution'] = 240
    logger.debug((model_name, metadata))
    register_model(
        MlfClipLoader(
            id=f"{model_name}--{dataset}",
            metadata=metadata),
    )