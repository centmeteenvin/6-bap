import logging
from transformers.pipelines import Conversation
from rag_chatbot.chatbot.chatbot import Chatbot
from openai import OpenAI

from rag_chatbot.secrets.secrets import Secrets

# Disable the http messages
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

class OpenAIChatbot(Chatbot):
    def __init__(self, modelName: str) -> None:
        super().__init__(modelName)
        self.modelName = modelName
        self.client = OpenAI(
            api_key=Secrets.openAIKey()
        )
        assert self.modelName in [model.id for model in self.client.models.list().data]

    def _askConversation(self, conversation: Conversation) -> Conversation:
        response = self.client.chat.completions.create(
            model = self.modelName,
            messages = conversation.messages
        )
        conversation.append_response(response.choices[0].message.content)
        return conversation
        