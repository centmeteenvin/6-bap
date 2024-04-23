from transformers import Conversation
from .chatbot import ChatBot
from abc import abstractmethod
from chatbot_domain import logger
from chatbot_domain.benchmark import WrongQuestion

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
        pass
    
    def _askQuestion(self, question: str) -> str:
        newQuestion = f"""
'ROLE': 
{self.getRole}
{question}
        """
        logger.debug(f"The added role was {self.getRole}")
        return self._chatbot._askQuestion(newQuestion)
    
    def __del__(self):
        del self._chatbot
    
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
    guardSentence = 'This question is outside my domain of knowledge.'
    
    @property
    def getRole(self) -> str:
        return f"""
    You are an assistant with expertise in the Image Processing domain. Your knowledge is given through the 'CONTEXT' section.
    You will try your best to answer the questions that have anything to do with the image processing domain.
    Some topics that are covered in this domain are:
        - The perception of light and the science behind it
        - sampling and quantization of images
        - Mathematical morphology
        - Filters for noise on images
    If relevant information to the question is given in the 'CONTEXT' section, you can safely assume it to be inside the domain.
    If the question is not remotely related to the domain you will answer with '{DIPDomainGuard.guardSentence}'
    """
    
class BenchmarkGuard(DomainGuard):
    """
    Primes the chatbot to work with Multiple answer questions
    """
    
    @property
    def getRole(self) -> str:
        return f"""
    You are undergoing a multiple choice examination. You will be asked a question followed by numbered options.
    The final format looks like this:
    'Reasoning: <Chain of thought>
     Answer: <Answer you chose>
     Option: <Number corresponding with the answer>'
     You must always answer using this format. Any other format will negatively effect your result.
     
     Be aware sometimes a question contains only wrong answers. When this happens, answer with the number corresponding with the '{WrongQuestion.answer}' option.
     At any time there is only one single correct answer.
     The options are numbered so the option section of your answer should only contain numbers.
    """