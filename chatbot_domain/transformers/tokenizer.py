from __future__ import annotations

from transformers import AutoTokenizer
from ..settings import Settings
from .. import logger
from datasets import DatasetDict, Dataset
class Tokenizer:
    _instance : None | "Tokenizer" = None
    
    def __init__(self) -> None:
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(Settings.modelName, token = Settings.accesToken, padding_side='left')
        self.max_length = None
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

    def get() -> "Tokenizer":
        if Tokenizer._instance is None:
            Tokenizer._instance = Tokenizer()
        return Tokenizer._instance
    
    def encode(self, dataSet: DatasetDict) -> DatasetDict:
        logger.info("tokenizing without max length")
        tokenizedDataSet =  dataSet.map(self._encode, batched=True)
        self._findAndSetMaxLength(tokenizedDataSet)
        logger.info("tokenizing with max length")
        logger.debug(f"batchSize: {Settings.batchSize}")
        return dataSet.map(self._encode, batched=True)
        
    def _encode(self, data: Dataset) -> Dataset:
        if self.max_length is None:
             tokenizedData =  self.tokenizer(data["sentence"], padding=True, truncation=True, return_tensors='pt')
        else:
             tokenizedData = self.tokenizer(data["sentence"], padding="max_length", truncation=True, return_tensors='pt', max_length=self.max_length)
        tokenizedData['labels'] = tokenizedData['input_ids'].clone()
        return tokenizedData
    
    def _findAndSetMaxLength(self, tokenizedData: DatasetDict) -> None:
        max_length_train = max([len(item['input_ids']) for item in tokenizedData['train']])
        max_length_evaluation = max([len(item['input_ids']) for item in tokenizedData['evaluation']]) 
        self.max_length = max(max_length_train, max_length_evaluation)
