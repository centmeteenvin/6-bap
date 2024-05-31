from rag_chatbot.similarity_search.text_search import TextSearch
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.text_source import TextSource


class DPRTextSearch(TextSearch):
    """
    This type of search uses the Facebook DPR model in combination with a Faiss index on the dataset to do similarity search.
    The DPR embeddings calculations are expensive so they are cached behind the screens. You can run the clear_cache script to reduce the cache.
    """
    def __init__(self, source: TextSource, cleanup: TextCleanup) -> None:
        super().__init__(source, cleanup)