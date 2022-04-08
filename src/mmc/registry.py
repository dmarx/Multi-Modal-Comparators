from loguru import logger

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .loaders.basemmcloader import BaseMmcLoader

class MmcRegistry:
    def __init__(self):
        self.loaders: List["BaseMmcLoader"] = []
    def find(
        self,
        **query
    ) -> List["BaseMmcLoader"]:
        """
        Searches the registry for MMCs loaders
        """
        hits = []
        for item in self.loaders:
            is_hit = True
            for k, v_query in query.items():
                v_item = getattr(item, k)
                if (v_item is not None) and (v_item != v_query):
                    is_hit = False
                    break
            if is_hit:
                hits.append(item)
        if len(hits) <1:
            logger.warning(f"No hits found for query: {query}")
        return hits

REGISTRY = MmcRegistry()

def register_model(mmc_loader: "BaseMmcLoader"):
    """
    Decorator that attaches mmc loaders to the REGISTRY 
    """
    logger.debug(f"registering model: {mmc_loader}")
    REGISTRY.loaders.append(mmc_loader)
    return mmc_loader