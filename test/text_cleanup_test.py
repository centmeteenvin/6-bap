from rag_chatbot.text_cleanup.implementations import AlphaNumericalTextTransformer, TrimTextTransformer
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup, TextTransformer


class ReplaceWithATextTransformer(TextTransformer):
    def transform(self, text: str) -> str:
        return "a"
    
class AppendBTextTransformer(TextTransformer):
    def transform(self, text: str) -> str:
       return text + "b"
    
def testReplaceWithATextTransformer():
    assert ReplaceWithATextTransformer().transform("foo") == "a"

def testAppendBTextTransformer():
    assert AppendBTextTransformer().transform("foo") == "foob"

def testTextCleanup():
    assert TextCleanup([ReplaceWithATextTransformer(), AppendBTextTransformer()]).process("foo") == "ab"
    assert TextCleanup([AppendBTextTransformer(), ReplaceWithATextTransformer()]).process("foo") == "a"

def testTextCleanupEntirePipeline():
    cleanup = TextCleanup([TrimTextTransformer(), AlphaNumericalTextTransformer()])
    assert cleanup.process('\n\n   foo\br2434$   ') == "foor2434$"