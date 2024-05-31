from rag_chatbot import NGramTextSearch, AlphaNumericalTextTransformer, TrimTextTransformer, TextCleanup, PDFTextSource
from rag_chatbot.cache.cache import clearCache, createCacheSubDir
import pathlib

textSource = PDFTextSource("C:\\Users\\vince\programmeren\projects\\6-bap\\test\\resources\\attention_is_all_you_need.pdf")
textCleanup = TextCleanup([TrimTextTransformer(), AlphaNumericalTextTransformer()])
textSearch = NGramTextSearch(textSource, textCleanup, 2)
results = textSearch.findNCClosest("Attention is a mechanism called transformer", 3)
print(results)
print(textSource.id)


resultingDir = createCacheSubDir(pathlib.Path("foo"))
print(resultingDir)
clearCache(needsConfirmation=True)