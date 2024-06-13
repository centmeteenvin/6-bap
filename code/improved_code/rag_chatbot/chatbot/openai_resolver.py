import logging
from typing import Generator
from transformers.pipelines import Conversation
from openai import OpenAI, Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from rag_chatbot.chatbot.resolver import StreamResolver
from rag_chatbot.secrets.secrets import Secrets

# Disable the http messages
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

class OpenAIResolver(StreamResolver):
    def __init__(self, modelName: str) -> None:
        super().__init__(modelName)
        self.modelName = modelName
        self.client = OpenAI(
            api_key=Secrets.openAIKey()
        )
        self.supportsSystemRole = True
        assert self.modelName in [model.id for model in self.client.models.list().data]

    def resolveConversation(self, conversation: Conversation) -> Conversation:
        response = self.client.chat.completions.create(
            model = self.modelName,
            messages = conversation.messages
        )
        conversation.append_response(response.choices[0].message.content)
        return conversation
    
    def _resolveStreamConversation(self, conversation: Conversation) -> Generator[str, None, None]:
        result = self.client.chat.completions.create(
            model= self.modelName,
            messages=conversation.messages,
            stream=True
        )
        result : Stream[ChatCompletionChunk]

        for chunk in result:
            yield chunk.choices[0].delta.content
        