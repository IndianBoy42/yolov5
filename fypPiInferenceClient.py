import json
import numpy as np
import os
import subprocess
from PIL import Image
import base64, json
import requests
import uuid
import websockets
import asyncio
from io import BytesIO
from azure import *

compute_server = "http://35.241.86.83:8000"
server = "http://192.168.137.192:12000"
# server = "http://172.31.175.255:12000"
# server = "http://192.168.45.171:12000"
# server = "http://175.159.124.105:12000"


def chunks(lst, n):
    while True:
        try:
            yield [next(lst) for i in range(n)]
        except:
            break


def getmac():
    mac = uuid.getnode()
    # return mac
    mac = f"{uuid.getnode():02x}"
    groups = ("".join(chunk) for chunk in chunks(iter(mac), 2))
    return "".join(groups)
    # return ':'.join(groups)


print(getmac())

source = "0"  # picamera
webcam = (
    source.isnumeric()
    or source.endswith(".txt")
    or source.lower().startswith(("rtsp://", "rtmp://", "http://"))
)
vid_path, vid_writer = None, None
shmstream = source.startswith("/tmp/")
if shmstream:
    source = f"shmsrc socket-path={source} \
            ! video/x-raw, format=BGR, width={int(imgsz_detect*4/3)}, height={imgsz_detect}, pixel-aspect-ratio=1/1, framerate=30/1 \
            ! decodebin \
            ! videoconvert \
            ! appsink"
    dataset = LoadStreamsBuffered(source, img_size=imgsz_detect)
elif webcam:
    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz_detect)
else:
    save_img = True
    dataset = LoadImages(source, img_size=imgsz_detect)


keepRetryingServerConnection = True

print("announceCamera")
while keepRetryingServerConnection:
    try:
        print("trying...")
        r = requests.post(
            server + "/setup/announceCamera",
            {
                "mac": getmac(),
                "isActive": "true",
            },
        )
        print(r)
        print(r.json())
        break
    except Exception as e:
        print(e)

dataset_iter = iter(dataset)
path, img, im0s, vid_cap = next(dataset_iter)
cv2.imwrite(f"{getmac()}.jpeg", im0s[0])
# print(im0s[0].shape)
# im = Image.fromarray(np.uint8(im0s[0].transpose(0, 3, 1, 2) * 255))
# with open(f"{getmac()}.jpeg", "wb") as f:
#     im.save(f, format="JPEG")

print("addCameraImage")
while keepRetryingServerConnection:
    try:  # Send the setupImage
        print("trying...")
        filename =  f"{getmac()}.jpeg"
        with open(filename, "rb") as f:
            r = requests.post(
                server + "/setup/addCameraImage",
                data={"mac": getmac(), "name": "name", "desc": "desc"},
                files={"file": (os.path.basename(filename), f, 'image/jpeg'),
                },
            )
            print("Response:", r)
            print("Res JSON:", r.json())
        break
    except Exception as e:
        print("addCameraImage Error:", e)

# async def listen():
#     url = "ws://192.168.45.227:12000"

#     async with websockets.connect(url) as ws:
#         while True: 
#             msg = await ws.recv()
#             print(msg)


# asyncio.get_event_loop().run_until_complete(listen())   
def remote_proc(img, im0s, view_img=False, **kwargs):
    res = []
    for im0 in im0s:
        if view_img: 
            cv2.imshow('view', im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        succ, buffer = cv2.imencode(".png", im0)
        f = BytesIO(buffer)
        r = requests.post(
            compute_server + "/lpr",
            files={"file": f},
        )
        res += r.json()['results']
    return res
        

prev = {}
for path, img, im0s, vid_cap in dataset_iter:
    res = remote_proc(img, im0s, view_img=view_img)
    
    if not webcam:
        input("Continue?")
    for detection in res:
        print("Curr: ", detection)
        if detection["lp"] in prev:  # Already detected
            continue
        print("New", detection)
        requests.put(
            server + "/operation/spotFilled", data={**detection, "mac": getmac()}
        )
    nxt = set(det["lp"] for det in res)
    for detection in prev:
        print("Old: ", detection)
        if detection in nxt:  # Still detected
            continue
        print("Gone", detection)
        requests.put(
            server + "/operation/spotVacated", data={"lp": detection, "mac": getmac()}
        )
    prev = nxt
