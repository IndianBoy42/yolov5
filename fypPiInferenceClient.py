import json
from contextlib import contextmanager
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
from helpers import *

compute_server = "http://35.241.86.83:8000"
server = "http://35.241.86.83:12000"
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


print("I am: ", getmac())

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
    view_img = False
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz_detect)
else:
    save_img = True
    dataset = LoadImages(source, img_size=imgsz_detect)


keepRetryingServerConnection = True


def announceCamera():
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


# capture and save one image for setup
dataset_iter = iter(dataset)
path, img, im0s, vid_cap = next(dataset_iter)
cv2.imwrite(f"{getmac()}.jpeg", im0s[0])
# print(im0s[0].shape)
# im = Image.fromarray(np.uint8(im0s[0].transpose(0, 3, 1, 2) * 255))
# with open(f"{getmac()}.jpeg", "wb") as f:
#     im.save(f, format="JPEG")


def addCameraImage():
    print("addCameraImage")
    while keepRetryingServerConnection:
        try:  # Send the setupImage
            print("trying...")
            filename = f"{getmac()}.jpeg"
            with open(filename, "rb") as f:
                r = requests.post(
                    server + "/setup/addCameraImage",
                    data={"mac": getmac(), "name": "name", "desc": "desc"},
                    files={
                        "file": (os.path.basename(filename), f, "image/jpeg"),
                    },
                )
                print("Response:", r)
                print("Res JSON:", r.json())
            break
        except Exception as e:
            print("addCameraImage Error:", e)


def remote_proc(img, im0s, view_img=False, **kwargs):
    res = []
    for im0 in im0s:
        if view_img:
            cv2.imshow("view", im0)
            if cv2.waitKey(1) == ord("q"):  # q to quit
                raise StopIteration
        succ, buffer = cv2.imencode(".png", im0)
        f = BytesIO(buffer)
        r = requests.post(
            compute_server + "/lpr",
            files={"file": f},
        )
        res += r.json()["results"]
    return res


def update_mem(l, arr, input):
    if len(arr) == l:
        for i in range(len(arr) - 1):
            arr[i] = arr[i + 1]
        arr[len(arr) - 1] = input
    else:
        arr.append(input)
    return arr


def processingIter(
    lprProc, mem, filled, buffSize, countThreshold, path, img, im0s, vid_cap
):
    res = lprProc(img, im0s, view_img=view_img)
    if not webcam:
        input("Continue?")
    for detection in res:
        print("Curr: ", detection)
        count = 0
        if detection["lp"] not in filled:
            for prev in mem:
                if detection["lp"] in prev:  # Already detected
                    count += 1
            if count == countThreshold:
                print("New", detection["lp"])
                filled.append(detection["lp"])
                serverResponse = requests.put(
                    server + "/operation/spotFilled",
                    data={**detection, "mac": getmac()},
                )
                print("Response:", serverResponse)

    # nxt = set(det["lp"] for det in res)
    print("Processing")
    print("Filled :", filled)
    mem = update_mem(buffSize, mem, set(det["lp"] for det in res))
    print("mem ", mem)
    finished = []
    for lp in filled:
        lpCount = 0
        # print("check license plate to remove: ",lp)
        for prev in mem:
            # print("prev: ", prev)
            isDetected = False
            for detection in prev:
                if lp == detection:
                    isDetected = True
                    break
            if isDetected:
                lpCount += 1
                # print("lpcount:" , lpCount)

        if lpCount < countThreshold:
            print("Gone", lp)
            requests.put(
                server + "/operation/spotVacated", data={"lp": lp, "mac": getmac()}
            )

        else:
            finished.append(lp)
            # print ("finished")
            # print(finished)

    filled = finished

    # for prev in mem:
    #     for detection in prev:
    #         print("Old: ", detection)
    #         if detection in nxt:  # Still detected
    #             continue
    #         print("Gone", detection)
    #         requests.put(
    #             server + "/operation/spotVacated", data={"lp": detection, "mac": getmac()}
    #         )


@contextmanager
def nopLock():
    try:
        yield True
    except e:
        pass
    finally:
        pass


def processingLoop(lock=None, lprProc=remote_proc):
    global dataset_iter
    # prev = {}
    mem = []
    filled = []
    buffSize = 5
    countThreshold = 3

    while True:
        if lock is not None:
            lock.acquire()
            lock.release()
        path, img, im0s, vid_cap = next(dataset_iter)
        processingIter(
            lprProc, mem, filled, buffSize, countThreshold, path, img, im0s, vid_cap
        )


# If this file is run directly
if __name__ == "__main__":
    announceCamera()
    addCameraImage()
    processingLoop()  # lprProc=proc for local inference
