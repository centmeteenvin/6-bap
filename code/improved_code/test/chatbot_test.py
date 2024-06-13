from typing import Any, Generator
import pytest
from transformers.pipelines import Conversation
from rag_chatbot.chatbot.anthropic_resolver import AnthropicResolver, AnthropicModels
from rag_chatbot.chatbot.augment import PromptAugment
from rag_chatbot.chatbot.chat_formatter import StreamChatFormatter
from rag_chatbot.chatbot.chatbot import Chatbot, StreamingChatbot
from shutil import get_terminal_size

from rag_chatbot.chatbot.openai_resolver import OpenAIResolver
from rag_chatbot.chatbot.rag_augment import RAGAugment
from rag_chatbot.chatbot.resolver import Resolver, StreamResolver
from rag_chatbot.similarity_search.text_search import QueryResult, TextSearch
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.text_source import Reference, TextSource


class FooResolver(Resolver):
    def __init__(self, name: str = "footloose") -> None:
        super().__init__(name)
        self.supportsSystemRole = True

    def resolveConversation(self, conversation: Conversation) -> Conversation:
        conversation.append_response("foo")
        return conversation


def test_chatbot_class(monkeypatch, capsys):
    with pytest.raises(AssertionError):
        Chatbot(resolver='')

    with pytest.raises(AssertionError):
        Chatbot(resolver=FooResolver)

    with pytest.raises(AssertionError):
        Chatbot(resolver = FooResolver(), augments=1)
    
    inputs = iter(["foo", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    resolver = FooResolver()
    chatbot = Chatbot(resolver=resolver)
    assert chatbot.askSingleQuestion("") == "foo"
    chatbot.startConversation()
    capture = capsys.readouterr()
    with open("./test/resources/conversation_output.txt", "w") as f:
        f.write(capture.out)
    assert (
        capture.out
        == f"""
{get_terminal_size().columns * '='}
Hello, you are now chatting with {resolver.name}. Type [exit] to quit the conversation.

{resolver.name}: foo
{get_terminal_size().columns * '-'}

That was a nice conversation.
{get_terminal_size().columns * '='}

"""
    )

class FooAugment(PromptAugment):
    def augmentConversation(self, conversation: Conversation) -> tuple[Conversation, Any]:
        conversation.add_message({'role': 'user', 'content': 'foo'})
        return conversation, 1

class BarAugment(PromptAugment):
    def augmentConversation(self, conversation: Conversation) -> tuple[Conversation, Any]:
        conversation.add_message({'role' : 'user', 'content': 'bar'})
        return conversation, 'bar'
        
def test_chatbot_prompt_augmentation():
    resolver = FooResolver()
    chatbotFooBar = Chatbot(resolver=resolver, augments=[FooAugment(), BarAugment()])

    emptyConversation = Conversation()
    fooBarConversation = chatbotFooBar._converse(emptyConversation)
    assert fooBarConversation[0]['content'] == 'foo'
    assert fooBarConversation[1]['content'] == 'bar'
    assert chatbotFooBar.result == [1, 'bar']
    emptyConversation = Conversation()
    chatbotBarFoo = Chatbot(resolver=resolver, augments=[BarAugment(), FooAugment()])
    barFooConversation = chatbotBarFoo._converse(emptyConversation)
    assert barFooConversation[0]['content'] == 'bar'
    assert barFooConversation[1]['content'] == 'foo'
    assert chatbotBarFoo.result == ['bar', 1]

def test_openAI_resolver():
    with pytest.raises(AssertionError):
        OpenAIResolver('foo')
    with pytest.raises(AssertionError):
        OpenAIResolver('footloose')
    resolver = OpenAIResolver('gpt-3.5-turbo-16k')
    conversation = Conversation("Hello from Belgium")
    conversation = resolver.resolveConversation(conversation)
    assert len(conversation.messages) == 2, "A new response should have been appended"
    response = conversation.messages[-1]
    assert response['role'] == 'assistant'
    assert len(response['content']) > 0

def test_anthropic_resolver():
    with pytest.raises(AssertionError):
        AnthropicResolver('foo')
    with pytest.raises(AssertionError):
        AnthropicResolver('footloose')
    resolver = AnthropicResolver(AnthropicModels.CLAUDE_3_HAIKU.value)
    conversation = Conversation('Hello from Belgium')
    conversation = resolver.resolveConversation(conversation)
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
        RAGAugment(None, FooResolver(), 1024)

    source = FooTextSource()
    cleanup = TextCleanup([])
    search = FooTextSearch(source, cleanup)
    with pytest.raises(AssertionError):
        RAGAugment(search, None, 1223)
    with pytest.raises(AssertionError):
        RAGAugment(search, FooResolver(), 0)

    augment = RAGAugment(search, FooResolver(),1024)
    conversation, results = augment.augmentConversation(Conversation("foo"))
    assert conversation.messages[-1]['role'] == 'user'
    assert conversation.messages[-1]['content'] == 'foo'
    assert conversation.messages[-2]['role'] == 'system'
    assert len(conversation.messages[-2]['content']) >= (1024 * 0.75)
    for result in results:
        assert isinstance(result, QueryResult)

class FooStreamingResolver(StreamResolver):
    def _resolveStreamConversation(self, conversation: Conversation) -> Generator[str, None, None]:
        for i in range(10):
            yield str(i)
    def resolveConversation(self, conversation: Conversation) -> Conversation:
        pass

def test_streaming_resolver():
    resolver = FooStreamingResolver('footloose')
    assert resolver.conversation is None
    resolved = resolver.resolveStreamConversation(Conversation())
    for result, expectation in zip(resolved, [str(i) for i in range(10)]):
        assert result == expectation
    assert resolver.conversation.messages[-1]['content'] == ''.join([str(i) for i in range(10)])

class FooFormatter(StreamChatFormatter):
    def __init__(self) -> None:
        super().__init__()
        self.SoC = False
        self.gQ = False
        self.PaR = False
        self.RR = False
        self.EoC = False
        self.RRS = False

    def startOfChat(self, name: str) -> None:
        self.SoC = True
        pass

    def getQuestion(self) -> str | None:
        if not self.gQ:
            self.gQ = True
            return ''
        return None

    def returnResponse(self, response: str) -> None:
        self.RR = True
        pass

    def processAdditionalResults(self, results: list) -> None:
        self.PaR = True
        pass

    def endOfChat(self) -> None:
        self.EoC = True
        pass

    def returnResponseStream(self, response: Generator[str, None, None]) -> None:
        assert isinstance(response, Generator)
        self.RRS = True
        for _ in response:
            pass
        pass

def test_streaming_chatbot_conversation():
    formatter = FooFormatter()
    with pytest.raises(AssertionError):
        StreamingChatbot(FooResolver(), formatter)

    chatbot = StreamingChatbot(FooStreamingResolver("footloose"), formatter)
    chatbot.streamConversation()
    assert formatter.SoC
    assert formatter.gQ
    assert not formatter.RR
    assert formatter.PaR
    assert formatter.EoC
    assert formatter.RRS
    