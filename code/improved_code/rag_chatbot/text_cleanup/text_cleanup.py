from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib

class TextTransformer(ABC):
    """A link in a cleanup chain"""
    @abstractmethod
    def transform(self, text:str) -> str:
        pass

    def id(self) -> str:
        """returns an unique identifier for this class. Is useful for discerning different text_source + cleanup pipelines from each other"""
        return hashlib.md5(self.__class__.__name__.encode()).hexdigest()

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
    
    def id(self) -> str:
        transformerString = ''
        for transformer in self.transformers:
            transformerString += transformer.id()
        return hashlib.md5(transformerString.encode()).hexdigest()