perceptor = MultiMMC(TEXT, IMAGE, shared_latent=True)

oa_clip_modelnames = [
  'RN50',
  'RN101',
  'ViTL64',
  ...
]

#perceptor.load_model(architecture='slip', id='some-clip-model')
#perceptor.load_model(architecture='blip', id='that-one-blip-model')

for model_name in oa_clip_modelnames:
  perceptor.load_model(
      architecture='clip', 
      publisher='openai', 
      id=model_name,
      #weight=1,
      )

# add a model that takes 50% responsibility for score 
perceptor.load_model(
    architecture='cloob', 
    publisher='crowsonkb', 
    weight=len(perceptor.models),
    )

logger.debug(perceptor.models.keys())

assert perceptor.supports_text
assert perceptor.supports_image
#assert perceptor.has_shared_latent

[m.name for m in perceptor.modalities]

import torch

text_container=["foo bar baz"]
image_container=img# torch.empty(1, 400,400,3)
multi_similarity_score = perceptor.compare(
    text=text_container, 
    image=image_container,
    return_projections = False,
    )

