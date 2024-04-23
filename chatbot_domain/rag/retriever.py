from abc import ABC, abstractmethod
from chatbot_domain import logger
from chatbot_domain.rag.dpr import DPR
from chatbot_domain.data import saveDataset
from datasets import Dataset


class Retriever(ABC):
    """
    An interface for retriever, they retrieve the most context relevant information of a knowledge base based on a question.
    """
    @abstractmethod
    def getContext(self, question: str, samples: int = 10) -> list[str]:
        """
        Returns a list with length samples which contain snippets relevant to question.
        """
        pass
    
class VectorRetriever(Retriever):
    """
    Uses A DPR model to encode a dataset and query it.
    """
    def __init__(self, dpr: DPR, dataset: Dataset, datasetLocation: str) -> None:
        """
        encode the dataset if necessary and adds Faiss index. the modified dataset will be saved to the given location.
        """
        super().__init__()
        self._dpr = dpr
        self._dataset =  dataset
        if 'embeddings' not in  dataset.column_names:
            self._encodeDataset()
        logger.info("Adding faiss Indices")
        self._dataset.add_faiss_index(column='embeddings')
        
        logger.info("Saving modified dataset")
        # saveDataset(self._dataset, datasetLocation, 'embeddings')
 
    def _encodeDataset(self) -> None:
        self._dataset = self._dataset.map(lambda example: {'embeddings': self._dpr.encodeContext(example['sentence'])})
        
    def getContext(self, question: str, samples: int = 10) -> list[str]:
        questionEmbedding = self._dpr.encodeQuestion(question)
        _, results = self._dataset.get_nearest_examples('embeddings', questionEmbedding, k=samples)
        return list(dict.fromkeys(results['paragraph'])) # Filter out duplicate paragraphs
        
