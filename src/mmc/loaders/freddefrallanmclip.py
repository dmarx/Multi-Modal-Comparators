
from loguru import logger
import napm
import torch

from .basemmcloader import BaseMmcLoader
#from .openaicliploader import OpenAiClipLoader
from ..modalities import TEXT, IMAGE
from ..multimodalcomparator import MultiModalComparator

from ..registry import REGISTRY, register_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def clip_from_model_name(model_name):
    model_name.split('_')
    return(model_name[0])

def mlang_from_model_name(model_name):
    model_name.split('_')
    return(model_name[1])

#class FrallanMClipLoader(OpenAiClipLoader):
class FrallanMClipLoader(BaseMmcLoader):
    """
    Multilingual CLIP loader
    * https://github.com/FreddeFrallan/Multilingual-CLIP
    * https://huggingface.co/M-CLIP
    """
    def __init__(
        self,
        id,
    ):
        self.architecture = 'clip'
        self.publisher = 'FreddeFrallan'
        self.id = id
        self.modalities = (TEXT, IMAGE)
    def load(self, device=DEVICE):
        #oai_ldr = super('RN50x4').load()
        #oai_model = oai_ldr.load()
        #im_model =oai_model.visual.clone()
        #del oai_model
        import clip
        clip_model = clip_from_model_name(self.id)
        model, preprocess_image = clip.load(clip_model, jit=False, device=device)
        model.eval()
        model.requires_grad_(False)
        #modelv = model.vision.clone()
        #modelv.eval()
        #modelv.requires_grad_(False)
        #modelv.to(device, memory_format=torch.channels_last)
        #del model
        modelv = model.vision

        logger.debug('using napm to "install" M-CLIP')
        url = "https://github.com/FreddeFrallan/Multilingual-CLIP"
        napm.pseudoinstall_git_repo(
            url, 
            env_name='mmc', 
            package_name='ff_multilingual_clip',
            add_install_dir_to_path=True,
        )
        napm.populate_pythonpaths('mmc')
        from ff_multilingual_clip.src import multilingual_clip
        mlang_model = mlang_from_model_name(self.id)
        text_model = multilingual_clip.load_model(mlang_model)
        #oai_model.modes['text']['projector'] = text_model
        #oai_model.modes['text']['preprocessor'] = 
        #oai_model.name = str(self)
        #return oai_model

        def preprocess_image_extended(*args, **kwargs):
            x = preprocess_image(*args, **kwargs)
            if x.ndim == 3:
                logger.debug("adding batch dimension")
                x = x.unsqueeze(0)
            return x
        mmc = MultiModalComparator(name=str(self), device=device)
        #mmc.register_modality(modality=TEXT, projector=model.encode_text, preprocessor=tokenizer)
        #mmc.register_modality(modality=IMAGE, projector=model.encode_image, preprocessor=preprocess_image_extended)
        mmc.register_modality(modality=TEXT, projector=text_model, preprocessor=None)
        mmc.register_modality(modality=IMAGE, projector=modelv, preprocessor=preprocess_image_extended)
        
        mmc._model = (text_model, modelv)
        return mmc

model_ids = [
    'RN50x4_M-BERT-Distil-40',
    'RN50x4_M-BERT-Distil-69',
    'RN50x4_Swe-CLIP-500k',
    'RN50x4_Swe-CLIP-2M',
    'ViT-B/32_M-BERT-Base-ViT-B'
]

for model in model_ids:
    register_model(
        FrallanMClipLoader(
            id=model
        )
    )