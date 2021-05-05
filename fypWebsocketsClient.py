import asyncio
import json
from fypPiInferenceClient import (
    addCameraImage,
    announceCamera,
    getmac,
    processingLoop,
    server,
    addCameraImage,
    announceCamera,
    processingLoop,
)
import websockets
import threading

TIMEOUT = 30


async def periodic():
    while True:
        await asyncio.sleep(1)


async def websockListener(uri):
    if uri.startswith("http"):
        uri = "ws" + uri.lstrip("http")

    processingLock = threading.Lock()
    processingLock.acquire()
    processingLocked = True
    processingThread = threading.Thread(target=processingLoop, args=(processingLock,))
    processingThread.start()

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps({"msg": "mac", "mac": getmac()}))

                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=TIMEOUT)
                    except Exception as e:
                        print(e)
                        break

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
                    elif message == "ping":
                        await websocket.send("pong")
        except Exception as e:
            await asyncio.sleep(1)
            print(e)


def startWebsockets(uri=server):
    announceCamera()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(websockListener(uri))


if __name__ == "__main__":
    startWebsockets()
