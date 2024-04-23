from abc import ABC, abstractmethod
from typing import final

from transformers import Conversation, pipeline

from chatbot_domain import logger
from chatbot_domain.rag.__main__ import Retriever
from chatbot_domain.settings import Settings
from chatbot_domain.transformers.model import Model
from chatbot_domain.transformers.tokenizer import Tokenizer
from shutil import get_terminal_size


class ChatBot(ABC):
    
    @property
    @abstractmethod
    def getName(self) -> str:
        """
        returns the models name to be displayed
        """    
    
    @abstractmethod
    def _askQuestion(self, question: str) -> str:
        """
        Implementation of askQuestion
        """
        pass
    
    @final
    def askQuestion(self, question: str) -> str:
        """
        Asks the question and returns an answer, no additional formatting is done on the answer.
        """
        logger.debug(f"The final prompt is {question}")
        return self._askQuestion(f"QUESTION:\n{question}")
    
    
    @abstractmethod
    def _askConversation(self, question: Conversation) -> Conversation:
        """
        see :func:`self.askConversation`
        """
        pass
    
    @final
    def askConversation(self, conversation: Conversation) -> Conversation:
        """
        Gives the conversational object to the chatbot and returns a conversation object with the response appended.
        """
        logger.debug("The final conversation object is:")
        return self._askConversation(conversation)

    
    def _representAnswer(self, answer: str) -> str:
        """
        Returns a string representation of the answer
        """
        return f"""
{get_terminal_size().columns * '-'}
{self.getName} : {answer}
{get_terminal_size().columns * '='}
    """
    
    @final
    def startConversation(self) -> None:
        """
        Starts an iterative prompting session
        The session will end when the user gives the "exit" input.
        """
        prompt : str = input("> ")
        conversation = Conversation(prompt)
        while prompt != "exit":
            response : Conversation = self._askConversation(conversation)
            print(self._representAnswer(response.messages[-1]['content']))
            prompt = input("> ")
            conversation.add_user_input(prompt)

class RAGChatBot(ChatBot):
    """
    This augments the normal ChatBot interface by prepending a context in relation to the question,
    it is used as a wrapper around another ChatBot object.
    """
    def __init__(self, chatbot : ChatBot,retriever: Retriever, contextLength = 512) -> None:
        super().__init__()
        self._chatbot = chatbot
        self._retriever = retriever
        self._contextLength = contextLength
        self._sampleGuess = 10 # A guess of the amount of samples to retrieve
    
    def _getContext(self,  question: str) -> str:
        """
        Returns context relevant to the question as a string
        """
        context = ""
        contextList = self._retriever.getContext(question, self._sampleGuess)
        for index, item in enumerate(contextList):
            if (len(context.split()) + len(item.split()) + len(question.split())) <= self._contextLength:
                context += '\n' + item
            else:
                self._sampleGuess = index + 1
                logger.debug(f"Had to cutoff the context due to the context window limit. updating sampleGuess to {self._sampleGuess}")
                return context
        else:
            self._sampleGuess = 2 * self._sampleGuess
            logger.debug(f"Context concatenation loop completed without breaking, updating sampleGuess to {self._sampleGuess}")
            logger.debug(f"rerunning context gathering with higher samples")
            return self._getContext(question)
        
    def __del__(self):
        del self._retriever
        del self._chatbot
        
    def _askConversation(self, conversation: Conversation) -> Conversation:
        """
        Prepends context to the current conversation object and appends the answer.
        """
        question = conversation.messages[-1]['content']
        context = self._getContext(question)
        conversation.add_message({
            'role': 'user',
            'content': f'CONTEXT: {context}'
        })
        logger.debug(f"The added context was {conversation}")
        return self._chatbot._askConversation(conversation)
    
    def _askQuestion(self, question: str) -> str:
        """
        Asks the Question and prepends the context to it.
        """
        context = self._getContext(question)
        newQuestion = f"""
'Context': 
{context}
{question}
        """
        logger.debug(f"The added context was {context}")
        return self._chatbot._askQuestion(newQuestion)
    
    @property
    def getName(self) -> str:
        return self._chatbot.getName + "+rag"
    
            
class ModelChatbot(ChatBot):
    def __init__(self, model: Model, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.tokenizer.padding_side = 'left'

        self.pipeline = pipeline("conversational", model= self.model.model, tokenizer=tokenizer.tokenizer)
    
    @property
    def getName(self) -> str:
        return self.model.name.split('/')[-1]
    
    def _askConversation(self, question: Conversation) -> Conversation:
        return self.pipeline([question], max_new_tokens = 1024)
    
    def _askQuestion(self, question: str) -> str:
        conversation = Conversation(question)
        responses : Conversation = self.pipeline([conversation], max_new_tokens = 1024)
        return responses.messages[-1]['content']
    
    def __del__(self):
        print("Delete called")
        del self.model
        del self.tokenizer
    
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


class OpenAIChatBot(ChatBot):
    """
    A chatbot using the openAI api.
    """
    
    def __init__(self, model: str = 'gpt-3.5-turbo') -> None:
        super().__init__()
        from chatbot_domain.secrets import OPEN_AI_API
        self.client = OpenAI(api_key=OPEN_AI_API)
        self._name = model
    
    @property
    def getName(self) -> str:
        return self._name
    
    def _askQuestion(self, question: str) -> str:
        result = self._askConversation(Conversation(question))
        return result.messages[-1]['content']
        
    def _askConversation(self, question: Conversation) -> Conversation:
        responses : ChatCompletion = self.client.chat.completions.create(
            model = self._name,
            messages= question.messages,
            stream=False
        )
        question.add_message(
            {key: value for key, value in responses.choices[0].message.model_dump().items() if key in ['content', 'role']}
            ) # Filter for only 'content' and 'role' keys
        return question
    

        