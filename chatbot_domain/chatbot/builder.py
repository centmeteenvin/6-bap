from __future__ import annotations
from .chatbot import ChatBot, RAGChatBot
from .guards import DomainGuard
from chatbot_domain.rag import Retriever
from typing import Type

class ChatBotBuilder():
    def __init__(self, base: ChatBot) -> None:
        self._base = base
        
    def rag(self, retriever : Retriever, contextLength : int = 512) -> "ChatBotBuilder":
        self._base = RAGChatBot(self._base, retriever, contextLength)
        return self
    
    def domainGuard(self, guard: Type[DomainGuard]) -> "ChatBotBuilder":
        self._base = guard(chatbot=self._base)
        return self
    
    def build(self) -> ChatBot:
        return self._base