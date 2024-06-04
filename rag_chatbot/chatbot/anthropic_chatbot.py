import logging
from anthropic import Anthropic
from transformers.pipelines import Conversation
from rag_chatbot.chatbot.chatbot import Chatbot
from enum import Enum

from rag_chatbot.secrets.secrets import Secrets

# Disable the http messages
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

class AnthropicModels(Enum):
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


class AnthropicChatbot(Chatbot):
    VALID_MODELS = [
        "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
    ]
    def __init__(self, modelName: str, maxTokens: int = 1024) -> None:
        super().__init__(modelName)
        assert modelName in self.VALID_MODELS
        assert isinstance(maxTokens, int) and maxTokens >= 0
        self.maxTokens = maxTokens
        self.client = Anthropic(api_key=Secrets.anthropicKey())
    
    def _askConversation(self, conversation: Conversation) -> Conversation:
        response = self.client.messages.create(
            model=self.name,
            messages = conversation.messages,
            max_tokens=self.maxTokens,
        )

        conversation.append_response(response.content[-1].text)
        return conversation