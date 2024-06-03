"""This file contains the abstract baseclass / interface of a chatbot"""

from abc import ABC, abstractmethod
from transformers.pipelines import Conversation
from shutil import get_terminal_size


class Chatbot(ABC):
    def __init__(self, name: str) -> None:
        super().__init__()
        assert isinstance(name, str), "The name must be a string"
        assert len(name) >= 4, "The name must at least contain 4 characters"
        self.name = name

    @abstractmethod
    def _askConversation(self, conversation: Conversation) -> Conversation:
        """Take a conversational input and return a conversational object with the response appended to it."""

    def askSingleQuestion(self, question: str) -> str:
        conversation = Conversation(question)
        return self._askConversation(conversation).messages[-1]["content"]

    def startConversation(self) -> str:
        print(
            f"""
{get_terminal_size().columns * '='}
Hello, you are now chatting with {self.name}. Type [exit] to quit the conversation."""
        )
        userInput = input("> ")
        conversation = Conversation(userInput)
        while userInput != "exit":
            conversation = self._askConversation(conversation)
            response = conversation.messages[-1]["content"]
            print(
                f"""
{self.name}: {response}
{get_terminal_size().columns * '-'}"""
            )
            userInput = input("> ")
            conversation.add_user_input(userInput)
        print(
            f"""
That was a nice conversation.
{get_terminal_size().columns * '='}
"""
        )
