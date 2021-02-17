import json
import numpy as np
import os
import subprocess
from PIL import Image
import base64, json
import requests
import uuid
from azure import *

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

for path, img, im0s, vid_cap in dataset:
    res = proc(img, im0s, view_img = view_img)
    print("Res", res)
