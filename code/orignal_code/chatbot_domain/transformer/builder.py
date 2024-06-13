from __future__ import annotations
from .model import Model, AdaptedModel
from .tokenizer import Tokenizer
class ModelBuilder():
    def __init__(self) -> None:
        self._modelName = None
        self._shouldQuantize = None
        self._deviceMap = None
        
    def modelName(self, model: str) -> ModelBuilder:
        self._modelName = model
        return self
    
    def shouldQuantize(self, shouldQuantize: bool) -> ModelBuilder:
        self._shouldQuantize = shouldQuantize
        return self
    
    def deviceMap(self, deviceMap : str) -> ModelBuilder:
        self._deviceMap = deviceMap
        return self
    
    def build(self, adapted: bool = False) ->tuple[Model, Tokenizer]:
        if adapted:
            return AdaptedModel(self._modelName, self._shouldQuantize, self._deviceMap), Tokenizer(self._modelName)
        return Model(self._modelName, self._shouldQuantize, self._deviceMap), Tokenizer(self._modelName)