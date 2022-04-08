import pytest 

def test_import_module():
    from mmc import registry

def test_import_REGISTRY():
    from mmc.registry import REGISTRY

def test_empty_search_REGISTRY():
    from mmc.registry import REGISTRY
    REGISTRY.find()

def test_clip_search_REGISTRY():
    from mmc.registry import REGISTRY
    hits = REGISTRY.find(architecture='clip')
    assert len(hits) > 1
