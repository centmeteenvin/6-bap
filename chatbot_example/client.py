import sys
import threading
from websockets.sync.client import connect, ClientConnection
isRunning = True

def receiveLoop(websocket: ClientConnection):
    global isRunning
    try:
        while isRunning:
            print(websocket.recv(), end= '', flush=True)
    except Exception as e:
        print(f"Receive the following exception, closing connection: {e}")
        isRunning = False

def inputLoop(websocket: ClientConnection):
    global isRunning
    try:
        while isRunning:
            userInput = input('')
            websocket.send(userInput)
    except Exception as e:
        print(f"Received the following error {e}, closing connection")
        isRunning = False


def main():
    uri = 'ws://localhost:1234'
    with connect(uri) as websocket:
        print("connected\n")
        receiveThread = threading.Thread(target=receiveLoop, args=[websocket])
        inputThread = threading.Thread(target=inputLoop, args=[websocket], daemon=True)
        receiveThread.start()
        inputThread.start()
        receiveThread.join()
        sys.exit()

if __name__ == "__main__":
    main()