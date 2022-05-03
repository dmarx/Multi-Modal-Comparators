# mmc

# installation

```
git clone https://github.com/dmarx/Multi-Modal-Comparators
cd 'Multi-Modal-Comparators'
pip install poetry
poetry build
pip install dist/mmc*.whl

# optional final step:
poe napm_installs
```

To see which models are immediately available, run:

```
python -m mmc.loaders
```

If you did not perform the optional `poe napm_installs` step, you likely received several warnings about 
models whose loaders could not be registered. These are models whose codebases depend on python code which
is not trivially installable. Skipping the optional last step will result in a faster installation. You will
still have access to all of the models supported by the library if you had run the last step, but they will
not be queryable and will need to be loaded using their mmc.loader directly. 

As a concrete example, the model `[cloob - corwsonkb - cloob_laion_400m_vit_b_16_32_epochs]` will not appear in the list of registered loaders, but can be loaded like this:

```
from mmc.loaders import KatCloobLoader

model = KatCloobLoader(id='cloob_laion_400m_vit_b_16_32_epochs').load()
```

Invoking the `load()` method on an unregistered loader will also invoke napm to prepare any uninstallable 
dependencies required to load the model. Next time you run `python -m mmc.loaders`, the CLOOB loader would
show as registered and no longer emit a warning for that model.

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



