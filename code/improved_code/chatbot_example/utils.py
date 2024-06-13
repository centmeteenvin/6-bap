"""Utility files"""
import json
from typing import Generator
import logging

from websockets.sync.server import ServerConnection
from rag_chatbot import StreamChatFormatter, QueryResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.debug("test if logger is printed")

class WebsocketChatFormatter(StreamChatFormatter):
    def __init__(self, socket: ServerConnection) -> None:
        """Pass the socket object to the formatter as it is its main IO source"""
        super().__init__()
        assert isinstance(socket, ServerConnection)
        self.socket = socket

    def startOfChat(self, name: str) -> None:
        message = f"""You are now chatting with {name}\n"""
        self.name = name
        self._send(message)


    def getQuestion(self) -> str | None:
        self._send("> ")
        question = self._recv()
        return None if question == 'exit' else question
    
    def returnResponseStream(self, response: Generator[str, None, None]) -> None:
        self._send(f"{self.name}: ")
        for chunk in response:
            self._send(chunk)
        self.socket.send('\n')
    
    def processAdditionalResults(self, results: list) -> None:
        for result in results:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, QueryResult):
                        item: QueryResult
                        self._send(json.dumps(item.reference))

    def returnResponse(self, response: str) -> None:
        self._send(f"{self.name}: {response}\n")

    def endOfChat(self) -> None:
        self._send("That was a really nice chat\n")
        self.socket.close()

    def _send(self, message: str) -> None:
        try:
            logger.debug(f"Sending message {message}")
            self.socket.send(message)
            logger.debug("Finished sending message")
        except Exception as e:
            logger.error(f"Got the following exception: {e}")

    def _recv(self) -> str:
        try:
            logger.debug("Awaiting input")
            result = self.socket.recv()
            logger.debug(f"Received: {result}")
            return result
        except Exception as e:
            logger.error(f"got the following exception: {e}")
            return "exit"