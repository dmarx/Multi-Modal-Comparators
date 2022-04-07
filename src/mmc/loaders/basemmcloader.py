from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..multimodalcomparator import MultiModalComparator

class BaseMmcLoader(ABC):
    """
    Base class that manages the procedure for loading MMC objects
    """
    def __init__(
        self,
        architecture=None,
        publisher=None,
        id=None,
    ):
        self.architecture = architecture
        self.publisher = publisher
        self.id = id
        self.modalities = ()
    @abc.abstractmethod
    def load(self) -> "MultiModalComparator":
        """
        Load the MMC object associated with this loader.
        """
        return

    def supports_modality(self, modality) -> bool:
        """
        Generic test for clarifying whether a specific modality is supported by the MMC this loader returns.
        """
        return any(modality.name == m.name for m in self.modalities)

    def __str__(self) -> str:
        return f"[{self.architecture} - {self.publisher} - {self.id}]"
