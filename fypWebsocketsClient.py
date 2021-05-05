import asyncio
import json
from fypPiInferenceClient import (
    addCameraImage,
    announceCamera,
    getmac,
    processingLoop,
)
from fypPiInferenceClient import addCameraImage, announceCamera, processingLoop
import websockets
import threading


async def websockListener(uri):
    if uri.startswith("http"):
        uri.lstrip("http")
        uri = "ws" + uri

    processingLock = threading.Lock()
    processingLock.acquire()
    processingLocked = True
    processingThread = threading.Thread(target=processingLoop, args=(processingLock,))
    processingThread.start()

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"msg": "mac", "mac": getmac()}))

        while True:
            message = await websocket.recv()

            print(f"< {message}")
            if message == "sendSetupImage":
                addCameraImage()
            elif message == "startLPR":
                if processingLocked:
                    processingLock.release()  # prevent the processing thread from acquiring the lock, thus it wont process
                    processingLocked = False
            elif message == "stopLPR":
                if not processingLocked:
                    processingLock.acquire()
                    processingLocked = True


def startWebsockets(uri="ws://35.241.86.83:12000"):
    announceCamera()

    asyncio.get_event_loop().run_until_complete(websockListener(uri))


if __name__ == "__main__":
    startWebsockets()
