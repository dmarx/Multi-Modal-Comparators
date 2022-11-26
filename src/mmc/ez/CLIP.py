import warnings

#from CLIP import clip as openai_clip
import clip as openai_clip

# initialize
#from ..loaders
import mmc
import mmc.loaders

#from ..mock.openai import MockOpenaiClip
#from ..registry import REGISTRY

from mmc.mock.openai import MockOpenaiClip
from mmc.registry import REGISTRY




class EzClip:
    def __init__(self):
        self._last_fetched_loader = None
        self._last_fetched_model = None
        self.d_openai={m.id:str(m) for m in REGISTRY.find(publisher='openai')}
    def _id_to_alias(self, id: str) -> str:
        """
        Converts a model id to an MMC alias.
        """
        if id in self.d_openai:
            return self.d_openai[id]
        else:
            return id

    def _alias_to_query(self, alias: str) -> dict:
        """
        Converts an MMC alias to a query.
        """
        architecture, publisher, id = alias[1:-1].split(' - ')
        return {'id':id, 'publisher':publisher, 'architecture':architecture}

    def available_models(self):
        """
        Returns a list of available models.
        """
        return list(self.d_openai.keys()) + [str(m) for m in REGISTRY.find()]

    def load(self, id, device=None):
        """
        Loads a model from the registry.
        """
        alias = self._id_to_alias(id)
        query = self._alias_to_query(alias)
        hits = REGISTRY.find(**query)
        if len(hits) < 1:
            raise ValueError(f"No model found for id: {id}")
        elif len(hits) > 1:
            raise ValueError(f"Multiple models found for id: {id}")

        loader = hits[0]
        model = loader.load(device)
        self._last_fetched_loader = loader
        self._last_fetched_model = model

        mocked = MockOpenaiClip(model, device)
        image_preprocessor = model
        return mocked, image_preprocessor

    @property
    def tokenize(self):
        """
        Returns the tokenizer for the last loaded model.
        """
        if self._last_fetched_model is None:
            #raise ValueError("No model loaded.")
            warnings.warn(
                "No model loaded. Returning OpenAI's default tokenizer. "
                "If this is not what you want, call `clip.load` before requesting the tokenizer."
            )
            return openai_clip.tokenize
        return self._last_fetched_model.modes['text']['preprocessor']


clip = EzClip()
