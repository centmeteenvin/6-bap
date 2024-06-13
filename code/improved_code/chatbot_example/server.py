import threading
from websockets.sync.server import serve, ServerConnection, WebSocketServer
from rag_chatbot import PDFTextSource, AlphaNumericalTextTransformer, TrimTextTransformer, TextCleanup, DPRTextSearch, StreamingChatbot, OpenAIResolver, RAGAugment
import logging

from .utils import WebsocketChatFormatter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

resolver = OpenAIResolver('gpt-3.5-turbo-16k')
logger.info("Connected with GPT 3.5")

source = PDFTextSource('./test/resources/attention_is_all_you_need.pdf')
cleanup = TextCleanup([TrimTextTransformer(), AlphaNumericalTextTransformer()])
search = DPRTextSearch(source, cleanup)
ragAugment = RAGAugment(search, resolver)
logger.info("Finished setting up similarity search")

def echo(websocket: ServerConnection):
    try:
        logger.info("A client has connected to the server")
        formatter = WebsocketChatFormatter(websocket)
        chatbot = StreamingChatbot(resolver=resolver, formatter=formatter, augments=[ragAugment])
        logger.info("Starting the conversation")
        chatbot.startConversation()
        logger.info("Conversation has ended")
    except Exception as e:
        logger.error(f"Got the following exception: {e}")
    websocket.close()
    return

def inputLoop(server: WebSocketServer) -> None:
    userInput = input('')
    while userInput != 'exit':
        userInput = input('')
    server.shutdown()


def main():
    with serve(echo, "localhost", 1234) as server:
        threading.Thread(target=inputLoop, args=[server]).start()
        server.serve_forever()

if __name__ == "__main__":
    main()