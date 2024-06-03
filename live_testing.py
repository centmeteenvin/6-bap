from openai import OpenAI

from rag_chatbot.chatbot.openai_chatbot import OpenAIChatbot
from rag_chatbot.secrets.secrets import Secrets

chatbot = OpenAIChatbot('gpt-3.5-turbo-16k')
chatbot.startConversation()