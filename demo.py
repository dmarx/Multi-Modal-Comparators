import torch

from mmc import MultiMMC
from mmc.modalities import TEXT, IMAGE


# for now at least, I'm referring to models like CLIP, CLOOB, SLIP etc. as 
# "multi-modal comparators" (MMCs). The MultiMMC class is a generic wrapper
# that serves the same function as like a "MultiCLIPPerceptor", but is
# intended to be suficiently generic to be able to wrap collections of models
# that aren't all from the same family. The conly constraint is that
# the individual MMCs attached to the MultiMMC must each be compatible with
# the modalities the MultiMMC supports.

perceptor = MultiMMC(TEXT, IMAGE, shared_latent=True)

oa_clip_modelnames = [
  'RN50',
  'RN101',
  'ViTL64',
  ...
]

#perceptor.load_model(architecture='slip', id='some-clip-model')
#perceptor.load_model(architecture='blip', id='that-one-blip-model')

# Individual MMCs can be ascribed weights. Potentially ways this could be used:
# * weighted ensemble of perceptors
# * compensate for perceptors that produce outputs at different scales
for model_name in oa_clip_modelnames:
    perceptor.load_model(
        architecture='clip', 
        publisher='openai', 
        id=model_name,
        #weight=1, # default
        )

# add a model that takes 50% responsibility for score, cause why not
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


text=["foo bar baz"]
image=IMAGE.read_from_disk('foobar.jpg')

multi_similarity_score = perceptor.compare(
    text=text_container, 
    image=image_container,
    return_projections = False,
    )

