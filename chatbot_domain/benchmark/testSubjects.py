import re
from abc import ABC, abstractmethod
from chatbot_domain import logger
from chatbot_domain.chatbot import ChatBot, DIPDomainGuard


class TestSubject(ABC):
    """This should be a proxy object. it takes a question and options and should return an 0 indexed integer answer corresponding to one of the options"""
    
    @abstractmethod
    def askQuestion(self, question: str, options: list[str]) -> int:
        pass
    
    @property
    @abstractmethod
    def getName(self) -> str:
        """Return a good name for the test-subject"""
        pass
        
class AlwaysATestSubject(TestSubject):
    def askQuestion(self, question: str, options: list[str]) -> int:
        return 0
    
    @property
    def getName(self) -> str:
        return "AlwaysA"
    
class NLPTestSubject(TestSubject):
    """Test subject for NLP Chatbots. Try to decorate your Chatbot object with the necessary guards"""
    def __init__(self, chatbot: ChatBot) -> None:
        super().__init__()
        self.chatbot = chatbot
        
    def askQuestion(self, question: str, options: list[str], recursion_level = 0) -> int:
        questionBuilder = "" + question + '\n'
        for option in options:
            questionBuilder += option
            questionBuilder += '\n'
        answer = self.chatbot.askQuestion(questionBuilder)
        # Check if the outside domain sentence was given.
        if answer.find(DIPDomainGuard.guardSentence) != -1:
            return len(options) - 1
        #Try to find the Option keyword.
        logger.debug(f"The answer the chatbot gave was {answer}")

        for line in reversed(answer.splitlines()):
            if "option" in line.lower():
                try:
                    matches = re.findall(r'(\d)[.,\s]*|$', line)
                    matches.pop()
                    option = int(matches[0])
                    break
                except IndexError as e:
                    logger.error(f"Got an index error meaning the option could not be extracted from the answer")
                    continue
        else:
            logger.error("Parsing loop completed without breaking.")
            if recursion_level < 3:
                logger.error("Reprompting the subject.")
                return self.askQuestion(question, options, recursion_level + 1)
            else:
                logger.error("Max recursion depth reached, will answer with option 4, wrong answers")
                return len(self.options) - 2
                    
        logger.debug(f"The option that was extracted: {option}")
        return min(option -1, 4)
    
    @property
    def getName(self) -> str:
        return self.chatbot.getName
        
    