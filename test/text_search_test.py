import pytest
from rag_chatbot.similarity_search.ngram_text_search import NGramTextSearch


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
