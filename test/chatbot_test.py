import pytest
from transformers.pipelines import Conversation
from rag_chatbot.chatbot.chatbot import Chatbot
from shutil import get_terminal_size

from rag_chatbot.chatbot.openai_chatbot import OpenAIChatbot


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
