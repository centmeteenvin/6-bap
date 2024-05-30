from abc import ABC, abstractmethod

class TextSource(ABC):
    """This is the abstract base class for every type of text source. Concrete implementations should overwrite the text property"""

    @abstractmethod
    @property
    def text(self):
        """This property returns the raw text of the text input"""
        pass