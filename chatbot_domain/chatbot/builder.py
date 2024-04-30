from __future__ import annotations
from .chatbot import ChatBot, RAGChatBot, OpenAIChatBot, ModelChatbot
from .guards import DomainGuard, BenchmarkGuard
from chatbot_domain.rag import Retriever, DPR, VectorRetriever
from chatbot_domain.data import parseData, createDataSet, loadDataSetFromDisk
from chatbot_domain.transformer import  Model, Tokenizer
from typing import Type
from datasets import Dataset

class _RetrieverPhaseRAGBuilder():
    """Will add the necessary DPR/encoding model to the dataset"""
    def __init__(self, dataset: Dataset, storePath: str) -> None:
        self._dataset = dataset
        self._storePath = storePath
    
    def vectorRetriever(self, dpr : DPR) -> VectorRetriever:
        """Adds the facebookDPR encoder to the builder"""
        return VectorRetriever(dpr, self._dataset, self._storePath)

class RAGBuilder():
    def __init__(self) -> None:
        self._retriever = None
    
    @staticmethod
    def fromFile(fileName: str, storePath: str | None = None) -> _RetrieverPhaseRAGBuilder:
        """The retriever will first parse a file into a dataset. if storePath is given the dataset will be stored there."""
        parsedData = parseData(fileName)
        dataset = createDataSet(parsedData, 0)
        return _RetrieverPhaseRAGBuilder(dataset, storePath)
        
        
    @staticmethod
    def fromDatasetDisk(loadPath: str) -> _RetrieverPhaseRAGBuilder:
        """Will load the dataset from the given path, additional columns will update the loadPath"""
        dataset = loadDataSetFromDisk(loadPath)
        return _RetrieverPhaseRAGBuilder(dataset, loadPath)
    
    @staticmethod
    def fromDataset(dataset: Dataset, storePath: str | None = None) -> _RetrieverPhaseRAGBuilder:
        """
        Will use the given dataset object.
        if storePath is given the dataset will be stored there.
        The object itself will be modified most likely
        """
        return _RetrieverPhaseRAGBuilder(dataset, storePath)
         

class ChatBotModifier():
    """Pass an existing chatbot model and it will decorate it to achieve additional behavior."""
    def __init__(self, base: ChatBot) -> None:
        self._base = base
        
    def rag(self, retriever : Retriever, contextLength : int = 512) -> "ChatBotModifier":
        """Adds the retriever to the system, consider using the RAGBuilder for this."""
        self._base = RAGChatBot(self._base, retriever, contextLength)
        return self
    
    def domainGuard(self, guard: Type[DomainGuard]) -> "ChatBotModifier":
        self._base = guard(chatbot=self._base)
        return self
    
    def benchmarkGuard(self) -> "ChatBotModifier":
        self._base = BenchmarkGuard(self._base)
        return self
    
    def build(self) -> ChatBot:
        return self._base
    

        

class ChatBotBuilder():
    @staticmethod
    def openAI(modelName: str | None = None) -> ChatBotModifier:
        "Builds an chatbot that works with openAI's API, default model will be GPT-3.5-turbo"
        if modelName is None:
            return ChatBotModifier(base=OpenAIChatBot())
        return ChatBotModifier(base=OpenAIChatBot(model=modelName))
    
    @staticmethod
    def model(modelAndTokenizer: tuple[Model, Tokenizer]) -> ChatBotModifier:
        "Returns a model builder which can be used to create the ModelChatBot, prefer using the .transformer ModelBuilder for this"
        return ChatBotModifier(base=ModelChatbot(modelAndTokenizer[0], modelAndTokenizer[1]))

    

        


        

