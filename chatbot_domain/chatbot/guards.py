from transformers import Conversation
from .chatbot import ChatBot
from abc import abstractmethod
from chatbot_domain import logger

class DomainGuard(ChatBot):
    """
    This is a decorator That adds an extra instruction before each question so that the assistant limits it's information to the domain.
    The subclasses implement the getRole property which will be inserted right before the question.
    """
    
    def __init__(self, chatbot : ChatBot) -> None:
        super().__init__()
        self._chatbot = chatbot
        
    @property
    def getName(self) -> str:
        return self._chatbot.getName + "+domainGuard"
    
    @property
    @abstractmethod
    def getRole(self) -> str:
        """ Returns the role of the assistant """
    
    def _askQuestion(self, question: str) -> str:
        newQuestion = f"""
'ROLE': 
{self.getRole}
{question}
        """
        logger.debug(f"The added role was {self.getRole}")
        return self._chatbot._askQuestion(newQuestion)
    
    def _askConversation(self, question: Conversation) -> Conversation:
        question.add_message({
            'role': 'user',
            'content': f"'ROLE':\n{self.getRole}"
        })
        logger.debug(f"The added role was {self.getRole}")
        return self._chatbot._askConversation(question)
    
class DIPDomainGuard(DomainGuard):
    """
    Adds the specific instruction to only answer answers that are available through the context
    """
    
    @property
    def getRole(self) -> str:
        return "You are an assistant with expertise in the Digital Image Processing (DIP) domain. Your knowledge is given through the 'CONTEXT' section. You will no answer any questions that are not related to the DIP domain. If you are not confident in your answer you will tell us so. If the question is not related to the domain you will answer with 'This question is out of my domain of knowledge'"