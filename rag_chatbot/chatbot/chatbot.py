"""This file contains the abstract baseclass / interface of a chatbot"""

from transformers.pipelines import Conversation
from shutil import get_terminal_size

from rag_chatbot.chatbot.augment import PromptAugment
from rag_chatbot.chatbot.resolver import Resolver


class Chatbot:
    def __init__(self, resolver: Resolver, augments: list[PromptAugment] = []) -> None:
        super().__init__()
        assert isinstance(
            resolver, Resolver
        ), f"The resolver must inherit from Resolver class, {type(resolver)}"
        assert isinstance(
            augments, list
        ), f"The augments must be a list not a {type(augments)}"
        for augment in augments:
            assert isinstance(
                augment, PromptAugment
            ), f"Each augment in the list must be a PromptAugment not a {type(augment)}"
        self.resolver = resolver
        self.augments = augments
        self.result = []
        """This holds the extra data that occurs when an Augment augments a prompt"""

    def _converse(self, conversation: Conversation) -> Conversation:
        """Puts the conversation through the pipeline of first augmenting it and
        the resolving it. The passed object will be modified"""
        self.result = []
        for augment in self.augments:
            conversation, result = augment.augmentConversation(conversation)
            self.result.append(result)
        return self.resolver.resolveConversation(conversation) # TODO implement prompt augments.

    def askSingleQuestion(self, question: str) -> str:
        conversation = Conversation(question)
        return self._converse(conversation).messages[-1]["content"]

    def startConversation(self) -> str:
        print(
            f"""
{get_terminal_size().columns * '='}
Hello, you are now chatting with {self.resolver.name}. Type [exit] to quit the conversation."""
        )
        userInput = input("> ")
        conversation = Conversation(userInput)
        while userInput != "exit":
            conversation = self._converse(conversation)
            response = conversation.messages[-1]["content"]
            print(
                f"""
{self.resolver.name}: {response}
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
