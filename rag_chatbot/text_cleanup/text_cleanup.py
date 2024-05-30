from abc import ABC, abstractmethod
from copy import deepcopy

class TextTransformer(ABC):
    """A link in a cleanup chain"""
    @abstractmethod
    def transform(self, text:str) -> str:
        pass

class TextCleanup():
    """
    This class takes a string and filters it so that it is presentable again.
    The transform method will apply all the given TextTransformers in sequence
    """
    def __init__(self, transformers: list[TextTransformer]) -> None:
        self.transformers = transformers
    
    def process(self, text: str) -> str:
        """Processes the given string through to each TextTransformer"""
        data = deepcopy(text)
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data