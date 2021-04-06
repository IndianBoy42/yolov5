import json
import numpy as np
import os
import subprocess
from PIL import Image
import base64, json
import requests
import uuid
from io import BytesIO
from azure import *

server = "http://192.168.45.171:12000"
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
    mac = f'{uuid.getnode():02x}'
    groups = (''.join(chunk) for chunk in chunks(iter(mac),2))
    return ':'.join(groups)

print(getmac())

source = '0' # picamera
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://'))
vid_path, vid_writer = None, None
shmstream = source.startswith('/tmp/')
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

print('announceCamera')
while keepRetryingServerConnection:
    try:
        print("trying...")
        r = requests.post(server + '/setup/announceCamera', {
            'mac': getmac(),
            'isActive': 'true',
        })
        print(r)
        print(r.json())
        break
    except Exception as e:
        print(e)

dataset_iter = iter(dataset)
path, img, im0s, vid_cap = next(dataset_iter)
im = Image.fromarray(np.uint8(im0s[0] * 255))
with open(f'{getmac()}.png', 'wb') as f:
    im.save(f, format='PNG')

    print('addCameraImage')
    while keepRetryingServerConnection:
        try: # Send the setupImage
            print("trying...")
            with open('firstImg.png', 'rb') as f:
                r = requests.post(server + '/setup/addCameraImage', data={
                    'mac': getmac(),
                    'name': "name",
                    'desc': "desc"
                }, files={
                    "image": f
                })
                print('Response:',r)
                print('Res JSON:', r.json())
            break
        except Exception as e:
            print('addCameraImage Error:', e)

init() # Initialize LPR


prev = {}
for path, img, im0s, vid_cap in dataset_iter:
    res = proc(img, im0s, view_img = view_img)
    if not webcam:
        input('Continue?')
    for detection in res:
        print('Curr: ', detection)
        if detection['lp'] in prev: # Already detected
            continue
        print('New')
        requests.put(server + '/operation/spotFilled', data={
            **detection,
            'mac': getmac()
        })
    nxt = set(det['lp'] for det in res)
    for detection in prev:
        print('Old: ', detection)
        if detection in nxt: # Still detected
            continue
        print('Gone')
        requests.put(server + '/operation/spotVacated', data={
            'lp': detection,
            'mac': getmac()
        })
    prev = nxt
