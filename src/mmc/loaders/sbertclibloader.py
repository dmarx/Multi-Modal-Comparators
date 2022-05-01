
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


class SBertClipLoader(BaseMmcLoader):
    """
    Multilingual text encoder aligned to the latent space of OpenAI's CLIP-ViT-B-32
    * https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1
    * https://www.sbert.net/docs/pretrained_models.html#image-text-models

    Primary language support: ar, bg, ca, cs, da, de, el, es, et, fa, fi, fr, fr-ca, 
    gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, 
    nl, pl, pt, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw

    Likely weak support for all languages compatible with multi-lingual DistillBERT:
    https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages
    """
    def __init__(
        self,
        #id,
    ):
        self.architecture = 'clip' # should this be a type too?
        self.publisher = 'sbert'
        self.id = 'ViT-B-32-multilingual-v1' #id
        self.modalities = (TEXT, IMAGE)
    def load(self, device=DEVICE):
        """
        Returns the MMC associated with this loader.
        """
        # TO DO: only load text encoder if the OpenAI CLIP image encoder is already 
        # attached to the invoking multimmc
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
        from sentence_transformers import SentenceTransformer
        img_model = SentenceTransformer('clip-ViT-B-32') # this should be identical to the model published by openai
        text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

        # default behavior returns numpy arrays. converting to tensors for API consistency
        def image_project_to_tensor(img):
            return torch.tensor(img_model.encode(img)).to(device)
        
        def text_project_to_tensor(txt):
            return torch.tensor(text_model.encode(txt)).to(device)

        # To do: we have a 'preprocess' pattern, should add a 'postprocess' pattern too. 
        # then instead of defining closures here, could just pass in TF.to_tensor()

        mmc = MultiModalComparator(name=str(self), device=device)
        mmc.register_modality(modality=TEXT, projector=text_project_to_tensor )#, preprocessor=tokenizer)
        mmc.register_modality(modality=IMAGE, projector=image_project_to_tensor )#, preprocessor=preprocess_image_extended)
        mmc._model = img_model
        return mmc

register_model(SBertClipLoader())