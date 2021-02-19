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

server = "http://192.168.116.227:12000"

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
    print(mac)
    groups = (''.join(chunk) for chunk in chunks(iter(mac),2))
    return ':'.join(groups)

source = '0' # picamera

init() # Initialize LPR

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

while True:
    try:
        r = requests.post(server + '/setup/announceCamera', {
            'mac': getmac(),
        })
        print(r)
        print(r.json())
        break
    except Exception as e:
        print(e)
dataset_iter = iter(dataset)
while True:
    try: # Send the setupImage
        path, img, im0s, vid_cap = next(dataset_iter)
        im = Image.fromarray(np.unit8(im0s * 255))
        with open('firstImg.jpg', 'rw') as f:
            im.save(f, format='JPEG')
            r = requests.post(server + '/setup/addCameraImage', {
                'mac': getmac(),
            }, files={
                "setupImage": f
            })
            print(r)
            print(r.json())
        break
    except Exception as e:
        print('addCameraImage Error:', e)

for path, img, im0s, vid_cap in dataset_iter:
    res = proc(img, im0s, view_img = view_img)
    print("Res", res)
    if not webcam:
        input('Continue?')
