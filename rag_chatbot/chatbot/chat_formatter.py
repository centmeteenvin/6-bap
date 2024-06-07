from __future__ import annotations
from abc import ABC, abstractmethod



class ChatFormatter(ABC):
    """
    This class contains several helper functions that print a chat to the
    screen. These can also be considered hooks to allow for more complex output
    than just the terminal. This means it also governs IO and such
    """

    @abstractmethod
    def startOfChat(self, name: str) -> None:
        """This indicates that a chat has started, the name will be the name of the resolver"""
        pass

    @abstractmethod
    def getQuestion(self) -> str | None:
        """
        Prompt the user for the question and return it as a string, if None
        is returned then the chatbot will interpret this as a request to exit
        """
        pass

    @abstractmethod
    def returnResponse(self, response: str) -> None:
        """Return the response to the user"""

    @abstractmethod
    def processAdditionalResults(self, results: list[any]) -> None:
        """This function is called after returnResponse. The additional result object from the promptAugments may be pass"""
        pass
    
    @abstractmethod
    def endOfChat(self) -> None:
        """This function is called when the chat has ended"""
        pass