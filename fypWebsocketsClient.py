import asyncio
from yolov5.fypPiInferenceClient import addCameraImage, announceCamera, processingLoop
import websockets
import threading


async def websockListener(uri):
    if uri.startswith("http"):
        uri.lstrip("http")
        uri = "ws" + uri

    processingLock = threading.Lock()
    processingLock.acquire()
    processingLocked = True
    processingThread = threading.Thread(target=processingLoop)
    processingThread.start()

    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()

            print(f"< {message}")
            if message == "sendSetupImage":
                addCameraImage()
            elif message == "startLPR":
                if not processingLocked:
                    processingLock.release()  # prevent the processing thread from acquiring the lock, thus it wont process
                    processingLocked = True
            elif message == "stopLPR":
                if processingLocked:
                    processingLock.acquire()
                    processingLocked = False


def startWebsockets(uri="ws://35.241.86.83:12000"):
    announceCamera()

    asyncio.get_event_loop().run_until_complete(websockListener(uri))