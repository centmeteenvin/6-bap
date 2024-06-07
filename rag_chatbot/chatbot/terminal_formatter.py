from shutil import get_terminal_size
from rag_chatbot.chatbot.chat_formatter import ChatFormatter
from rag_chatbot.similarity_search.text_search import QueryResult


class TerminalFormatter(ChatFormatter):
    """A simple formatter that uses the terminal as the main IO source"""

    def startOfChat(self, name: str) -> None:
        self.name = name
        print(
            f"""
{get_terminal_size().columns * '='}
Hello, you are now chatting with {name}. Type [exit] to quit the conversation."""
        )

    def getQuestion(self) -> str | None:
        question = input("> ")
        return None if question == "exit" else question

    def returnResponse(self, response: str) -> None:
        print(
            f"""
{self.name}: {response}
{get_terminal_size().columns * '-'}"""
        )

    def processAdditionalResults(self, results: list) -> None:
        if len(results) == 0:
            return # No interesting information was given.
        for result in results:
            if isinstance(result, list):
                if all([isinstance(subResult, QueryResult) for subResult in result]):
                    #This means we found a list of RAG query results
                    for queryResult in result:
                        queryResult : QueryResult
                        print(f"reference: {queryResult.reference}")


    def endOfChat(self) -> None:
        print(
            f"""
That was a nice conversation.
{get_terminal_size().columns * '='}
"""
        )
