from rag_chatbot.text_cleanup.implementations import AlphaNumericalTextTransformer, TrimTextTransformer
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup, TextTransformer
import pathlib

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

def testTransformerId():
    hashPaths = pathlib.Path("./test/resources/cleanup_hashes")
    if not hashPaths.exists():
        hashPaths.mkdir(parents=True)
    textCleanup1 = TextCleanup([AppendBTextTransformer(), ReplaceWithATextTransformer()])
    textCleanup2 = TextCleanup([ReplaceWithATextTransformer(), AppendBTextTransformer()])
    if len(list(hashPaths.iterdir())) == 0:
        with open(hashPaths / pathlib.Path("replace_with_a_text_transformer_hash.txt"), 'w') as f:
            f.write(ReplaceWithATextTransformer().id())
            f.flush()
        with open(hashPaths / pathlib.Path("append_b_text_transformer_hash.txt"), 'w') as f:
            f.write(AppendBTextTransformer().id())
            f.flush()
        with open(hashPaths / "textCleanup1.txt", 'w') as f:
            f.write(textCleanup1.id())
        with open(hashPaths / "textCleanup2.txt", 'w') as f:
            f.write(textCleanup2.id())
        assert False, "This test should rerun as it was not primed correctly"
    
    with open(hashPaths / pathlib.Path("replace_with_a_text_transformer_hash.txt"), 'r') as f:
        replaceWithAId = f.read()
    with open(hashPaths / pathlib.Path("append_b_text_transformer_hash.txt"), 'r') as f:
        appendBId = f.read()
    with open(hashPaths / "textCleanup1.txt", 'r') as f:
        textCleanup1Id = f.read()
    with open(hashPaths / "textCleanup2.txt", 'r') as f:
        textCleanup2Id = f.read()

    assert replaceWithAId == ReplaceWithATextTransformer().id()
    assert appendBId == AppendBTextTransformer().id()
    assert appendBId != replaceWithAId
    assert textCleanup1.id() == textCleanup1Id
    assert textCleanup2.id() == textCleanup2Id
    assert textCleanup2Id != textCleanup1Id
