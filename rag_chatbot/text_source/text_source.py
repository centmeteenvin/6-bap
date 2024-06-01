from abc import ABC, abstractmethod
from typing import Hashable

class Reference(ABC):
    """This abstract class provides a reference for each possible text excerpt"""
    @abstractmethod
    def get(self) -> str:
        """Returns the string representation of the source"""
        pass
    
class TextSource(ABC):
    """This is the abstract base class for every type of text source. Concrete implementations should overwrite the text property"""

    @property
    @abstractmethod
    def text(self) -> list[tuple[str, Reference]]:
        """This property returns the raw text of the text input as a list of tuples"""
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """Returns a unique identifier of the object, is needed for caching expensive operations"""
        pass