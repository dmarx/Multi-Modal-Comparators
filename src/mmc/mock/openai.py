"""
API Mocks for models published by OpenAI
"""

import torch
from dataclasses import dataclass

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from loguru import logger


class MockOpenaiClipModule:
    """
    Mocks the OpenAI CLIP.clip module API
    """
    def __init__(self, loader, device=DEVICE):
        self._loader = loader
        self.device = device

    def available_models(self):
        # ...should this return the mmc registry?
        return str(self._loader)

    @property
    def _model(self):
        if not hasattr(self, '_model_'):
            self._model_ = self._loader.load(self.device)
        return self._model_

    @property
    def tokenize(self):
        return self._model.modes['text']['preprocessor']

    @property
    def preprocess_image(self):
        return self._model.modes['image']['preprocessor']

    @property
    def load(self, id, device=DEVICE):
        clip = MockOpenaiClip(self._model, self.device)
        return clip, self.preprocess_image


@dataclass
class MockVisionModel:
    input_resolution: int = 224
    output_dim: int = 1024


class MockOpenaiClip:
    """
    Wrapper class to facilitate drop-in replacement with MMC models where 
    model interface conforms to OpenAI's CLIP implementationare.
    """
    def __init__(
        self,
        mmc_object,
        device=DEVICE,
    ):
        assert mmc_object.supports_text
        assert mmc_object.supports_image

        #if (mmc_object.publisher == 'openai') and (mmc_object.architecture == 'clip'):
        #    return mmc_object._model

        self.device = device
        self.mmc_object = mmc_object

        vision_args = {}
        if hasattr(mmc_object, 'input_resolution'):
            vision_args['input_resolution'] = mmc_object.input_resolution
        elif hasattr(mmc_object, 'metadata'):
            vision_args['input_resolution'] = mmc_object.metadata.get(
                'input_resolution', 
                vision_args['input_resolution']
                )

        if hasattr(mmc_object, '_model'):
            if hasattr(mmc_object._model, 'visual'):
                if hasattr(mmc_object._model.visual, 'input_resolution'):
                    self.visual = mmc_object._model.visual
                elif hasattr(mmc_object._model.visual, 'image_size'):
                    self.visual = mmc_object._model.visual
                    self.visual.input_resolution = self.visual.image_size

        if not hasattr(self, 'visual'):
            logger.debug("'visual' attribute not found in model. Mocking vision model API.")
            logger.debug(vision_args)
            self.visual = MockVisionModel(**vision_args)
    
    def encode_image(
        self, 
        image: torch.Tensor,
    ) -> torch.Tensor:
        #return self.mmc_object.project_image(image)
        # bypass pre-processor
        #project = self.mmc_object.modes['image']['projector']
        #return project(image)
        return self.mmc_object.project_image(image, preprocess=False)

    def encode_text(
        self,
        text: torch.Tensor,
    ) -> torch.Tensor:
        #return self.mmc_object.project_text(text)
        # bypass pre-processor
        #project = self.mmc_object.modes['text']['projector']
        #return project(text)
        return self.mmc_object.project_text(text, preprocess=False)
