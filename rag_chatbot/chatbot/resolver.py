from abc import ABC, abstractmethod
from transformers.pipelines import Conversation

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