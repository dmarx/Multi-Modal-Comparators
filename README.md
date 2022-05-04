# mmc

# installation

```
git clone https://github.com/dmarx/Multi-Modal-Comparators
cd 'Multi-Modal-Comparators'
pip install poetry
poetry build
pip install dist/mmc*.whl

# optional final step:
#poe napm_installs
python src/mmc/napm_installs/__init__.py
```

To see which models are immediately available, run:

```
python -m mmc.loaders
```

### That optional `poe napm_installs` step

For the most convenient experience, it is recommended that you perform the final `poe napm_installs` step. 
Omitting this step will make your one-time setup faster, but will make certain use cases more complex.

If you did not perform the optional `poe napm_installs` step, you likely received several warnings about 
models whose loaders could not be registered. These are models whose codebases depend on python code which
is not trivially installable. You will still have access to all of the models supported by the library as if 
you had run the last step, but their loaders will not be queryable from the registry (see below) and will need 
to be loaded via the appropriate mmc.loader directly, which may be non-trivial to identify without the ability to 
query it from mmc's registry. 

As a concrete example, if the napm step is skipped, the model `[cloob - corwsonkb - cloob_laion_400m_vit_b_16_32_epochs]` 
will not appear in the list of registered loaders, but can still be loaded like this:

```
from mmc.loaders import KatCloobLoader

model = KatCloobLoader(id='cloob_laion_400m_vit_b_16_32_epochs').load()
```

Invoking the `load()` method on an unregistered loader will invoke [napm](https://github.com/dmarx/not-a-package-manager) 
to prepare any uninstallable dependencies required to load the model. Next time you run `python -m mmc.loaders`, 
the CLOOB loader will show as registered and spinning up the registry will longer emit a warning for that model.


# Usage

**TLDR**

```
# spin up the registry
from mmc import loaders

from mmc.mock.openai import MockOpenaiClip
from mmc.registry import REGISTRY

cloob_query = {architecture='cloob'}
cloob_loaders = REGISTRY.find(**cloob_query)

# loader repl prints attributes for uniquely querying
print(cloob_loaders)

# loader returns a perceptor whose API is standardized across mmc
cloob_model = cloob_loaders[0].load()

# wrapper classes are provided for mocking popular implementations
# to facilitate drop-in compatibility with existing code
drop_in_replacement__cloob_model = MockOpenaiClip(cloob_model)
```

## Querying the Model Registry

Spin up the model registry by importing the loaders module:

```from mmc import loaders```

To see which models are available:

```
from mmc.registry import REGISTRY

for loader in REGISTRY.find():
    print(loader)
```

You can constrain the result set by querying the registry for specific metadata attributes

```
# all CLIP models
clip_loaders = REGISTRY.find(architecture='clip')

# CLIP models published by openai
openai_clip_loaders = REGISTRY.find(architecture='clip', publisher='openai')

# All models published by MLFoundations (openCLIP)
mlf_loaders = REGISTRY.find(publisher='mlfoundations)'

# A specific model
rn50_loader = REGISTRY.find(architecture='clip', publisher='openai', id='RN50')
# NB: there may be multiple models matching a particular "id". the 'id' field
# only needs to be unique for a given architecture-publisher pair.
```

All pretrained checkpoints are uniquely identifiable by a combination of `architecture`, `publisher`, and `id`. 

The above queries return lists of **loader** objects. If model artifacts (checkpoints, config) need to be downloaded, they will only be downloaded after the `load()` method on the loader is invoked. 

```
loaders = REGISTRY.find(...)
loader = loaders[0] # just picking an arbitrary return value here, remember: loaders is a *list* of loaders
model = loader.load()
```

The `load()` method returns an instance of an `mmc.MultiModalComparator`. The `MultiModalComparator` class
is a modality-agnostic abstraction. I'll get to the ins and outs of that another time.

## API Mocking

You want something you can just drop into your code and it'll work. We got you. This library provides wrapper
classes to mock the APIs of commonly used CLIP implementations. To wrap a `MultiModalComparator` so it can
be used as a drop-in replacement with code compatible with OpenAI's CLIP:

```
from mmc.mock.openai import MockOpenaiClip

my_model = my_model_loader.load()
model = MockOpenaiClip(my_model)
```

## MultiMMC: Multi-Perceptor Implementation

The `MultiMMC` class can be used to run inference against multiple mmc models in parallel. This form of 
ensemble is sometimes referred to as a "multi-perceptor".

To ensure that all models loaded into the MultiMMC are compatible, the MultiMMC instance is initialized
by specifying the modalities it supports. We'll discuss modality objects in a bit.

```
from mmc.multimmc import MultiMMC
from mmc.modalities import TEXT, IMAGE

perceptor = MultiMMC(TEXT, IMAGE)
```

To load and use a model:

```
perceptor.load_model(
    architecture='clip', 
    publisher='openai', 
    id='RN50',
)

score = perceptor.compare(
    image=PIL.Image.open(...), 
    text=text_pos),
)
```

Additional models can be added to the ensemble via the `load_model()` method.

The MultiMMC does not support API mocking because of its reliance on the `compare` method.


# Available Pre-trained Models

```
# [<architecture> - <publisher> - <id>]
[clip - openai - RN50]
[clip - openai - RN101]
[clip - openai - RN50x4]
[clip - openai - RN50x16]
[clip - openai - RN50x64]
[clip - openai - ViT-B/32]
[clip - openai - ViT-B/16]
[clip - openai - ViT-L/14]
[clip - openai - ViT-L/14@336px]
[clip - mlfoundations - RN50--openai]
[clip - mlfoundations - RN50--yfcc15m]
[clip - mlfoundations - RN50--cc12m]
[clip - mlfoundations - RN50-quickgelu--openai]
[clip - mlfoundations - RN50-quickgelu--yfcc15m]
[clip - mlfoundations - RN50-quickgelu--cc12m]
[clip - mlfoundations - RN101--openai]
[clip - mlfoundations - RN101--yfcc15m]
[clip - mlfoundations - RN101-quickgelu--openai]
[clip - mlfoundations - RN101-quickgelu--yfcc15m]
[clip - mlfoundations - RN50x4--openai]
[clip - mlfoundations - RN50x16--openai]
[clip - mlfoundations - ViT-B-32--openai]
[clip - mlfoundations - ViT-B-32--laion400m_e31]
[clip - mlfoundations - ViT-B-32--laion400m_e32]
[clip - mlfoundations - ViT-B-32--laion400m_avg]
[clip - mlfoundations - ViT-B-32-quickgelu--openai]
[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e31]
[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e32]
[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_avg]
[clip - mlfoundations - ViT-B-16--openai]
[clip - mlfoundations - ViT-L-14--openai]
[clip - sbert - ViT-B-32-multilingual-v1]
[clip - sajjjadayobi - clipfa]

# The following models depend on napm for setup
[clip - navervision - kelip_ViT-B/32]
[cloob - crowsonkb - cloob_laion_400m_vit_b_16_16_epochs]
[cloob - crowsonkb - cloob_laion_400m_vit_b_16_32_epochs]
[clip - facebookresearch - clip_small_25ep]
[clip - facebookresearch - clip_base_25ep]
[clip - facebookresearch - clip_large_25ep]
[slip - facebookresearch - slip_small_25ep]
[slip - facebookresearch - slip_small_50ep]
[slip - facebookresearch - slip_small_100ep]
[slip - facebookresearch - slip_base_25ep]
[slip - facebookresearch - slip_base_50ep]
[slip - facebookresearch - slip_base_100ep]
[slip - facebookresearch - slip_large_25ep]
[slip - facebookresearch - slip_large_50ep]
[slip - facebookresearch - slip_large_100ep]
[simclr - facebookresearch - simclr_small_25ep]
[simclr - facebookresearch - simclr_base_25ep]
[simclr - facebookresearch - simclr_large_25ep]
[clip - facebookresearch - clip_base_cc3m_40ep]
[clip - facebookresearch - clip_base_cc12m_35ep]
[slip - facebookresearch - slip_base_cc3m_40ep]
[slip - facebookresearch - slip_base_cc12m_35ep]
```

# Contributing

## Suggest a pre-trained model

If you would like to suggest a pre-trained model for future addition, you can add a comment to [this issue](https://github.com/dmarx/Multi-Modal-Comparators/issues/2)

## Add a pre-trained model

1. Create a loader class that encapsulates the logic for importing the model, loading weights, preprocessing inputs, and performing projections. 
2. At the bottom of the file defining the loader class should be a code snippet that adds each respective checkpoint's loader to the registry.
3. Add an import for the new file to `mmc/loaders/__init__.py`. The imports in this file are the reason `import mmc.loaders` "spins up" the registry.
4. If the codebase on which the model depends can be installed, update `pytproject.toml` to install it.
5. Otherwise, add napm preparation at the top of the loaders `load` method (see cloob or kelip for examples), and also add napm setup to `mmc/napm_installs/__init__.py`
6. Add a test case to tests/test_mmc_loaders.py
7. Add a test script for the loader (see `test_mmc_katcloob` as an example)


