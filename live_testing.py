from rag_chatbot.similarity_search.ngram_text_search import NGramTextSearch
from rag_chatbot.text_cleanup.implementations import AlphaNumericalTextTransformer, TrimTextTransformer
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.pdf_text_source import PDFTextSource

textSource = PDFTextSource("C:\\Users\\vince\programmeren\projects\\6-bap\\test\\resources\\attention_is_all_you_need.pdf")
textCleanup = TextCleanup([TrimTextTransformer(), AlphaNumericalTextTransformer()])
textSearch = NGramTextSearch(textSource, textCleanup, 2)
results = textSearch.findNCClosest("Attention is a mechanism called transformer", 3)
print(results)