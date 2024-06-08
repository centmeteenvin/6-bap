from abc import ABC, abstractmethod
from typing import Generator
from transformers.pipelines import Conversation

from rag_chatbot.chatbot.types import Stream

class Resolver(ABC):
    """A resolver's job is the resolve an answer from a Conversation object. That is it nothing more"""
    def __init__(self, name: str) -> None:
        super().__init__()
        assert isinstance(name, str), "The name must be a string"
        assert len(name) >= 4, "The name must at least contain 4 characters"
        self.name = name
        self.supportsSystemRole = False
        """Set this to True if the resolver allows for the system role in a conversation."""

    @abstractmethod
    def resolveConversation(self, conversation: Conversation) -> Conversation:
        """
        Take the conversational object and resolve it. The answer is always
        appended at the end of the Conversation object. The passed conversation
        and the returned conversation are the same object. This means that the
        passed conversation object is augmented.
        """
        pass

class StreamResolver(Resolver):
    """This class can also resolve a conversation and return a string answer"""
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.conversation : Conversation = None
    
    def resolveStreamConversation(self, conversation: Conversation) -> Generator[str, None, Conversation]:
        """
        The reduce overhead for implementations we only require them to emit
        a normal stream of strings, the conversational object on the other hand
        will be handled by this function, this function collects the streams
        output and re-yields it. In the meantime if will also concatenate all
        the bits internally and finally return the modified conversational
        object. This function will also put the result in self.conversation.
        """
        stream = self._resolveStreamConversation(conversation)
        answer = ''
        for text in stream:
            answer += text
            yield text

        conversation.append_response(answer)
        self.conversation = conversation
        return self.conversation

    @abstractmethod
    def _resolveStreamConversation(self, conversation: Conversation) -> Stream:
        """
        This function should return a stream of str values, the concatenation
        without separators should return the complete answer
        """
        pass