from abc import ABC, abstractmethod
from typing import Any
from transformers.pipelines import Conversation
class PromptAugment(ABC):
    """These classes contain a function to augment a conversational object"""

    @abstractmethod
    def augmentConversation(self, conversation: Conversation) -> tuple[Conversation, Any]:
        """
        This augments a conversation and returns a modified conversation and
        space for augment results like references in the case of RAG
        """

        