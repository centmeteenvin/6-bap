"""This file contains the abstract baseclass / interface of a chatbot"""

from typing import Generator
from transformers.pipelines import Conversation

from rag_chatbot.chatbot.augment import PromptAugment
from rag_chatbot.chatbot.chat_formatter import ChatFormatter, StreamChatFormatter
from rag_chatbot.chatbot.resolver import Resolver, StreamResolver
from rag_chatbot.chatbot.terminal_formatter import TerminalFormatter


class Chatbot:
    def __init__(
        self,
        resolver: Resolver,
        formatter: ChatFormatter = TerminalFormatter(),
        augments: list[PromptAugment] = [],
    ) -> None:
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

        assert isinstance(formatter, ChatFormatter)
        self.resolver = resolver
        self.augments = augments
        self.formatter = formatter
        self.result = []
        """This holds the extra data that occurs when an Augment augments a prompt"""


    def _augment(self, conversation: Conversation) -> Conversation:
        """
        Put the conversational object through to augmentation pipeline, the
        passed conversation will be modified, additional results will be
        available in self.result
        """
        self.result = []
        for augment in self.augments:
            conversation, result = augment.augmentConversation(conversation)
            if self.result is not None:
                self.result.append(result)
        return conversation

    def _converse(self, conversation: Conversation) -> Conversation:
        """Puts the conversation through the pipeline of first augmenting it and
        the resolving it. The passed object will be modified"""
        conversation = self._augment(conversation)
        return self.resolver.resolveConversation(
            conversation
        )

    def askSingleQuestion(self, question: str) -> str:
        conversation = Conversation(question)
        return self._converse(conversation).messages[-1]["content"]

    def startConversation(self) -> str:
        self.formatter.startOfChat(self.resolver.name)
        userInput = self.formatter.getQuestion()
        conversation = Conversation(userInput)
        while userInput is not None:
            conversation = self._converse(conversation)
            response = conversation.messages[-1]["content"]
            self.formatter.returnResponse(response)
            self.formatter.processAdditionalResults(self.result)

            userInput = self.formatter.getQuestion()
            conversation.add_user_input(userInput)
        self.formatter.endOfChat()


class StreamingChatbot(Chatbot):
    """
    A chatbot that has adds streaming capabilities. It requires a more strenuous
    requirement for the compositions. The streaming capability in question is
    that the answer will now be a generator.
    """

    def __init__(
        self,
        resolver: StreamResolver,
        formatter: StreamChatFormatter = TerminalFormatter(),
        augments: list[PromptAugment] = [],
    ) -> None:
        super().__init__(resolver, formatter, augments)
        assert isinstance(resolver, StreamResolver)
        assert isinstance(formatter, StreamChatFormatter)
        self.formatter : StreamChatFormatter
        self.resolver : StreamResolver

    def streamConversation(self) -> None:
        """The same as self.startConversation but now the self.formatter.returnResponseStream function will return the output"""
        self.formatter.startOfChat(self.resolver.name)
        userInput = self.formatter.getQuestion()
        conversation = Conversation(userInput)
        while userInput is not None:
            conversation = self._augment(conversation)
            stream = self.resolver.resolveStreamConversation(conversation)
            self.formatter.returnResponseStream(stream)
            conversation = self.resolver.conversation
            self.formatter.processAdditionalResults(self.result)
            userInput = self.formatter.getQuestion()
            conversation.add_user_input(userInput)
        self.formatter.endOfChat()

    def streamSingleQuestion(self, question: str) -> Generator[str, None, None]:
        """Same as single question but returns a generator now."""
        conversation = Conversation(question)
        conversation = self._augment(conversation)
        return self.resolver.resolveStreamConversation(conversation)
