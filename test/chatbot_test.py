import pytest
from transformers.pipelines import Conversation
from rag_chatbot.chatbot.anthropic_chatbot import AnthropicChatbot, AnthropicModels
from rag_chatbot.chatbot.chatbot import Chatbot
from shutil import get_terminal_size

from rag_chatbot.chatbot.openai_chatbot import OpenAIChatbot
from rag_chatbot.chatbot.rag_augment import RAGAugment
from rag_chatbot.similarity_search.text_search import QueryResult, TextSearch
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.text_source import Reference, TextSource


class FooChatbot(Chatbot):
    def __init__(self, name: str = "footloose") -> None:
        super().__init__(name)

    def _askConversation(self, conversation: Conversation) -> Conversation:
        conversation.append_response("foo")
        return conversation


def test_chatbot_class(monkeypatch, capsys):
    with pytest.raises(AssertionError):
        FooChatbot(name=1564654)

    with pytest.raises(AssertionError):
        FooChatbot(name="foo")

    inputs = iter(["foo", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    chatbot = FooChatbot()
    assert chatbot.askSingleQuestion("") == "foo"
    chatbot.startConversation()
    capture = capsys.readouterr()
    with open("./test/resources/conversation_output.txt", "w") as f:
        f.write(capture.out)
    assert (
        capture.out
        == f"""
{get_terminal_size().columns * '='}
Hello, you are now chatting with {chatbot.name}. Type [exit] to quit the conversation.

footloose: foo
{get_terminal_size().columns * '-'}

That was a nice conversation.
{get_terminal_size().columns * '='}

"""
    )

def test_openAI_chatbot():
    with pytest.raises(AssertionError):
        OpenAIChatbot('foo')
    with pytest.raises(AssertionError):
        OpenAIChatbot('footloose')
    chatbot = OpenAIChatbot('gpt-3.5-turbo-16k')
    conversation = Conversation("Hello from Belgium")
    conversation = chatbot._askConversation(conversation)
    assert len(conversation.messages) == 2, "A new response should have been appended"
    response = conversation.messages[-1]
    assert response['role'] == 'assistant'
    assert len(response['content']) > 0

def test_anthropic_chatbot():
    with pytest.raises(AssertionError):
        AnthropicChatbot('foo')
    with pytest.raises(AssertionError):
        AnthropicChatbot('footloose')
    chatbot = AnthropicChatbot(AnthropicModels.CLAUDE_3_HAIKU.value)
    conversation = Conversation('Hello from Belgium')
    conversation = chatbot._askConversation(conversation)
    assert len(conversation.messages) == 2, "A new response should have been appended"
    response = conversation.messages[-1]
    assert response['role'] == 'assistant'
    assert len(response['content']) > 0

class FooTextSource(TextSource):
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def text(self) -> str:
        return ''
    
    @property
    def id(self) -> str:
        return ''
class FooReference(Reference):
    @property
    def get(self) -> str:
        return ''
    
    def asDict(self) -> str:
        return {}
    

class FooTextSearch(TextSearch):
    def __init__(self, source: TextSource, cleanup: TextCleanup) -> None:
        super().__init__(source, cleanup)

    def findNCClosest(self, query: str, n: int) -> list[QueryResult]:
        return [QueryResult(query, FooReference(), 0) for _ in range(n)]

def test_ragAugment():
    with pytest.raises(AssertionError):
        RAGAugment(None, 1024)

    source = FooTextSource()
    cleanup = TextCleanup([])
    search = FooTextSearch(source, cleanup)
    with pytest.raises(AssertionError):
        RAGAugment(search, 0)

    augment = RAGAugment(search, 1024)
    conversation, results = augment.augmentPrompt(Conversation("foo"))
    assert conversation.messages[-1]['role'] == 'user'
    assert conversation.messages[-1]['content'] == 'foo'
    assert conversation.messages[-2]['role'] == 'system'
    assert len(conversation.messages[-2]['content']) >= (1024 * 0.75)
    for result in results:
        assert isinstance(result, QueryResult)