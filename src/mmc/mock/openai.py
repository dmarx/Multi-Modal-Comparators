"""
API Mocks for models published by OpenAI
"""

from turtle import width
import torch
from dataclasses import dataclass

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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

        self.vision = MockVisionModel(**vision_args)
    
    def encode_image(
        self, 
        image: torch.Tensor,
    ) -> torch.Tensor:
        return self.mmc_object.project_image(image)

    def encode_text(
        self,
        text: torch.Tensor,
    ) -> torch.Tensor:
        return self.mmc_object.project_text(text)