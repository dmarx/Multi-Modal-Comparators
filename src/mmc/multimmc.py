"""
Utilities for facilitating computing similarity scores via ensembles of MMCs,
which must be at least compatible with respect to the two modes being compared.
"""

from loguru import logger

from .registry import REGISTRY

class MultiMMC(MultiModalComparator):
    _registry = REGISTRY #?
    def __init__(
        self, 
        *modalities, 
        device=DEVICE,
    ):
        self.device=device
        self.modalities = modalities
        self.models = {}
    # probably shouldn't need to redefine/override this method
    def supports_modality(self, modality):
        return any(modality.name == m.name for m in self.modalities)
    # probably shouldn't need to redefine/override this method
    def _supports_mode(self, modality_name):
        return any(modality_name == m.name for m in self.modalities) #self.modes
    def load_model(
        self,
        architecture='clip', 
        publisher='openai', 
        id=None,
        weight=1,
        device=None,
    ):
        if device is None:
            device = self.device
        model_loaders = self._registry.find(architecture=architecture, publisher=publisher, id=id)
        for model_loader in model_loaders:
            assert all(model_loader.supports_modality(m) for m in self.modalities)
            model_key = f"[{architecture} - {publisher} - {id}]"
            if model_key not in self.models:
                model = model_loader.load()
                self.models[model_key] = {'model':model, 'weight':weight}
                #self.models[model_key] = {'model':model.to(device), 'weight':weight}
            else:
                logger.warning(f"Model already loaded: {model_key}")

    def _project_item(self, item, mode) -> dict:
        assert self._supports_mode(mode)
        projections = {}
        for model_name, d_model in self.models.items():
            model, weight = d_model['model'], d_model['weight']
            logger.debug(model_name)
            logger.debug(model)
            if model._supports_mode(mode):
                #item.to(model.device)
                projections[model.name] = {'projection':model._project_item(item, mode), 'weight':weight}
        return {'modality':mode, 'projections':projections}

    def compare(
        self,
        return_projections = False,
        **kwargs,
    ):
        projections = {} # d['modality']['model_key'] = vector
        for modality_name, item in kwargs.items():
            projections[modality_name] = self._project_item(item, modality_name)
        outv = self._reduce_projections(projections)
        if return_projections:
            outv = (outv, projections)
        return outv
    def _reduce_projections(self, projections, return_raw_scores=False):
        accumulator = 0
        raw_scores = {}
        # this is hideous and should be trivially vectorizable. 
        for model_key, d_perceptor in self.models.items():
            perceptor = d_perceptor['model']
            weight = d_perceptor['weight']

            # compute per-model scores and compute weighted sum
            kargs = {}
            for modality_name, d_vectors in projections:
                kargs[modality_name] = d_vectors[model_key]
            score = perceptor._reduce_projections(**kargs)
            accumulator += score * weight

            if return_raw_scores:
                raw_scores[model_key] = score

        outv = accumulator
        if return_raw_scores:
            outv = (accumulator, raw_scores)
        return outv
    