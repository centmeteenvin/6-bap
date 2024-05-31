import re
import string
from rag_chatbot.similarity_search.text_search import QueryResult
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.text_source import TextSource
from .text_search import TextSearch


class NGramTextSearch(TextSearch):
    def __init__(self, source: TextSource, cleanup: TextCleanup, n: int) -> None:
        super().__init__(source, cleanup)
        assert isinstance(n, int) and n > 0, "n must be a number larger then 0"
        self.n = n

    @staticmethod
    def nGram(source: str, n: int) -> list[str]:
        """Generate all the possible n-grams of the source string. n should larger or equal to 1"""
        assert n >= 1, "N must be larger then or equal to 1"
        words = source.split()
        assert len(words) >= n, "The amount of words should be greater then or equal to n"
        nGrams = []
        for i in range(len(words)-n + 1):
            nGrams.append(' '.join(words[i:i+n]))
        return nGrams
    
    @staticmethod
    def retainAlphaNumerical(text: str) -> str:
        pattern = f"[^{re.escape(string.ascii_letters + string.digits + string.whitespace)}]"
        return re.sub(pattern, '', text).lower()
    
    @staticmethod
    def calculateNGramScore(query: str, data: str, n: int) -> int:
        """This function calculates the n-gram score for every piece of text available. Both strings are preprocessed by removing none-alpha numerical character"""
        cleanQuery = NGramTextSearch.retainAlphaNumerical(query)
        cleanData = NGramTextSearch.retainAlphaNumerical(data)
        queryNGram = NGramTextSearch.nGram(cleanQuery, n)
        dataNGram = NGramTextSearch.nGram(cleanData, n)
        score = 0
        for nGram in queryNGram:
            score += dataNGram.count(nGram)
        return score
    
    def findNCClosest(self, query: str, n: int) -> list[QueryResult]:
        if len(self.retainAlphaNumerical(query).split()) < self.n:
            raise Exception(f"The amount of words in the query must be greater than {self.n}")
        results = []
        for text, reference in self.text:
            results.append(
                QueryResult(text, reference, score=float(self.calculateNGramScore(query, text, self.n)))
            ) 
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:n]
