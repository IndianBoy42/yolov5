# import the necessary packages
import cv2
import argparse
import socket
import time
import shutil
from pathlib import Path
import requests
import os

# https://stackoverflow.com/questions/46219454/how-to-open-a-gstreamer-pipeline-from-opencv-with-videowriter

def post(fn, data):
    requests.post(args.server + '/' + fn, data)

def post_lpr(line):
    linedata = line.split(' ')
    labels = ['number', '2lines', 'x', 'y', 'w', 'h']
    data = dict(zip(labels, linedata))

    print(data)

    # data['number'], data['2lines'], data['x'], data['y'], data['w'], data['h'] = line.split(' ')

    # post('lpr', data)

def gstreamer_rtp(fps = 30, frame_width=2133, frame_height=1600):
    gst_str = f"appsrc \
        ! videoconvert \
        ! shmsink socket-path={args.file} shm-size=100000000 sync=true wait-for-connection=false"
    print(gst_str)

    out = cv2.VideoWriter(gst_str, 0, fps, (frame_width, frame_height), True)
    if not out.isOpened():
        raise ValueError('Not Opened')
    
    print("Starting Sending Loop")
    old = set(Path(dir_labels).iterdir())
    i = 0
    while True:
        # read the frame from the file and send it to the 'out' stream
        f = input('next? ')
        if f == 'q': break
        # f = 'inference/input/2.JPG' # Hardcode for testing

        if ';' in f:
            fs = [x.strip() for x in f.split(';')]
        else:
            fs = [f]


        for f in fs:
            frame = cv2.imread(f)
            if frame is None:
                continue
            print(f, frame.shape)
    
            frame = cv2.resize(frame, (frame_width,frame_height))
    
            out.write(frame)

        while True:
            new = set(Path(dir_labels).iterdir())
            dif = new.difference(old)
            if len(dif) != 0:
                print(dif)
                for f in dif: 
                    for l in f.read_text().split('\n'):
                        l = l.strip()
                        if len(l) == 0: continue
                        print(l)
                        post_lpr(l)
                break
            old = new
            time.sleep(5)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", default='/tmp/shm',
    help="the name of the shared memory file (usually /tmp/*)")
ap.add_argument("-d", "--dir_yolo", default='runs/demo',
    help="Where will yolo output the data?")
ap.add_argument("-s", "--server", default='127.0.0.1:8080',
    help="Server for storing this data")
    
args = ap.parse_args()
print(args)

dir_labels = Path(args.dir_yolo) / 'labels'
if dir_labels.exists():
    shutil.rmtree(dir_labels)
dir_labels.mkdir(parents=True, exist_ok=True)
print(dir_labels)

gstreamer_rtp()
        

