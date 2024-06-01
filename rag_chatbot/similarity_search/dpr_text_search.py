from rag_chatbot.similarity_search.text_search import TextSearch
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.text_source import TextSource
import numpy

class DPREncoder():
    """This class contains two encoders, one to encode a question into embeddings and one for the context"""

    def __init__(self) -> None:
        """Loads the necessary encoders. One thing that is needed is that the hugging face token is passed"""
        
    def encode(self, data: str) -> numpy.ndarray:
        

class DPRTextSearch(TextSearch):
    """
    This type of search uses the Facebook DPR model in combination with a Faiss index on the dataset to do similarity search.
    The DPR embeddings calculations are expensive so they are cached behind the screens. You can run the clear_cache script to reduce the cache.
    """
    def __init__(self, source: TextSource, cleanup: TextCleanup) -> None:
        super().__init__(source, cleanup)