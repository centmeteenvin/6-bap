from anthropic import Anthropic

from rag_chatbot.chatbot.anthropic_chatbot import AnthropicChatbot, AnthropicModels
from rag_chatbot.secrets.secrets import Secrets
print(AnthropicModels.CLAUDE_3_HAIKU.value)
chatbot = AnthropicChatbot(AnthropicModels.CLAUDE_3_HAIKU.value)
chatbot.startConversation()