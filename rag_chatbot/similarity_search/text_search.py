from abc import ABC, abstractmethod

from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source import TextSource, Reference

class QueryResult:
    def __init__(self, text: str, reference: Reference, score: float) -> None:
        self.text = text
        self.reference = reference
        self.score = score

    def __repr__(self) -> str:
        return f"""
Text: 
{self.text[0:255]}...
Reference: 
{self.reference}
Score: {self.score}
        """

class TextSearch(ABC):
    """This class takes a string input and it's job is to find the closest related piece of text"""

    def __init__(self, source: TextSource, cleanup: TextCleanup) -> None:
        super().__init__()
        self.source = source
        self.cleanup = cleanup
        self._textCache = None

    @property
    def text(self) -> list[tuple[str, Reference]]:
        if self._textCache is None:
            self._textCache = []
            for text, reference in self.source.text:
                processedText = self.cleanup.process(text)
                self._textCache.append((processedText, reference))
        return self._textCache

    @abstractmethod
    def findNCClosest(self, query: str, n: int) -> list[QueryResult]:
        """Find the N most similar to the query pieces of text. If N is greater than the amount of pieces available the result will be truncated"""
        pass