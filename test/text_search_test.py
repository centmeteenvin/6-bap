import pytest
import torch
from rag_chatbot.cache.cache import MODULE_CACHE_DIR, clearCache
from rag_chatbot.similarity_search.dpr_text_search import DPREncoder, DPRTextSearch
from rag_chatbot.similarity_search.ngram_text_search import NGramTextSearch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

from rag_chatbot.similarity_search.text_search import QueryResult
from rag_chatbot.text_cleanup.implementations import AlphaNumericalTextTransformer, TrimTextTransformer
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.pdf_text_source import PDFTextSource


def test_n_gram_generation():
    source = "I love Luna Very Much"
    assert NGramTextSearch.nGram(source, 2) == ["I love", "love Luna", "Luna Very", "Very Much"]
    assert NGramTextSearch.nGram(source,3) == ["I love Luna", "love Luna Very", "Luna Very Much"]
    assert NGramTextSearch.nGram(source, 1) == ["I", "love", "Luna", "Very", "Much"]
    shortSource = "I love Luna"
    assert NGramTextSearch.nGram(shortSource, 3) == ["I love Luna"]
    tooShortSource = "I love"
    with pytest.raises(AssertionError):
        NGramTextSearch.nGram(tooShortSource, 3)
    with pytest.raises(AssertionError):
        NGramTextSearch.nGram(tooShortSource, 0)
    
def test_retain_alpha_numerical():
    test = " Hello $ world.'!"
    assert NGramTextSearch.retainAlphaNumerical(test) == " hello  world"
    assert NGramTextSearch.retainAlphaNumerical(test).split() == ["hello", "world"]

def test_n_gram_score():
    query = "foo"
    data = "bar bar foo"
    assert NGramTextSearch.calculateNGramScore(query, data, 1) == 1

    query = "foo bar"
    data = "foo bar foo bar"
    assert NGramTextSearch.calculateNGramScore(query, data, 1) == 4
    assert NGramTextSearch.calculateNGramScore(data, query, 1) == 4
    assert NGramTextSearch.calculateNGramScore(query, data, 2) == 2

def test_DPR_encoding():
    encoder = DPREncoder()
    assert isinstance(encoder.ctxTokenizer, DPRContextEncoderTokenizerFast)
    assert isinstance(encoder.ctxEncoder, DPRContextEncoder)
    assert isinstance(encoder.questionTokenizer, DPRQuestionEncoderTokenizerFast)
    assert isinstance(encoder.questionEncoder, DPRQuestionEncoder)
    data = "Hello world"
    contextEncoding = encoder.encodeContext(data)
    assert isinstance(contextEncoding, torch.Tensor)
    assert (encoder.encodeContext(data) == contextEncoding).all(), "The encoding is not consistent between equal data"
    questionEncoding = encoder.encodeQuestion(data)
    assert isinstance(questionEncoding, torch.Tensor)
    assert (encoder.encodeQuestion(data) == questionEncoding).all(), "The encoding is not consistent between equal data"
    assert (questionEncoding != contextEncoding).all(), "The question and context encoding was the same"

@pytest.fixture()
def cacheClear():
    clearCache()
    yield
    clearCache()

def test_text_source_DPR_Encoding(cacheClear):
    source = PDFTextSource("./test/resources/attention_is_all_you_need.pdf")
    cleanup = TextCleanup([TrimTextTransformer(), AlphaNumericalTextTransformer()])
    search = DPRTextSearch(source, cleanup)
    assert set(search.dataset.column_names) == {"text", "reference"}
    assert set(search.embeddingsDataset.column_names) == {"sentence", "embeddings", "refId"}
    assert search.embeddingsDataset.list_indexes() == ["embeddings"], "Faiss index was not added"
    cacheFiles = [path.name for path in search.cache_path.iterdir()]
    assert 'data.set' in cacheFiles
    assert 'embeddings.set' in cacheFiles
    assert 'embeddings.index' in cacheFiles
    assert len(list(MODULE_CACHE_DIR.iterdir())) != 0

def test_text_source_findNClosest(cacheClear):
    source = PDFTextSource("./test/resources/attention_is_all_you_need.pdf")
    cleanup = TextCleanup([TrimTextTransformer(), AlphaNumericalTextTransformer()])
    search = DPRTextSearch(source, cleanup)
    with pytest.raises(AssertionError):
        search.findNCClosest("fdsgfdg", 10000)
    
    result = search.findNCClosest("foo bar", 3)
    assert len(result) == 3
    for item in result:
        assert isinstance(item, QueryResult)
    