# import the necessary packages
import imutils.video as imv
import cv2
import imagezmq
import argparse
import socket
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", default='/tmp/shm',
    help="the name of the shared memory file (usually /tmp/*)")
ap.add_argument('--cap', action='store_true', help='save images')  
args = ap.parse_args()

print(args)

def outloop(out, fw, fh):
    print("Starting Sending Loop")
    i = 0
    while True:
        # read the frame from the file and send it to the 'out' stream
        f = input('next? ')
        if f == 'q': break
        f = 'inference/input/2.JPG'

        frame = cv2.imread(f)
        frame = cv2.resize(frame, (fw,fh))
        print(f, frame.shape)
        # frame = cv2.imread(input("fname: "))
                
        out(frame)
        i += 1
        print('.', end='')
        if i % 10 == 0:
            print()

def caploop(cap):
    print("Starting Sending Loop")
    
    i = 0
    while True:
        ret, frame = cap.read()

        print(frame.shape)

        cv2.imwrite('output.jpg', frame)
        
        i += 1
        print('.')
        if i % 10 == 0:
            print()

# https://stackoverflow.com/questions/46219454/how-to-open-a-gstreamer-pipeline-from-opencv-with-videowriter

def gstreamer_rtp(fps = 30, frame_width=2133, frame_height=1600):
    gst_str = f"appsrc \
        ! videoconvert \
        ! shmsink socket-path={args.file} shm-size=100000000 sync=true wait-for-connection=false"
    print(gst_str)

    out = cv2.VideoWriter(gst_str, 0, fps, (frame_width, frame_height), True)
    if not out.isOpened():
        raise ValueError('Not Opened')
    
    outloop(out.write, frame_width, frame_height)

def gstreamer_rtp_cap(fps = 30, frame_width=2133, frame_height=1600):
    gst_str = f"shmsrc socket-path={args.file} \
        ! video/x-raw, format=BGR, width={frame_width}, height={frame_height}, pixel-aspect-ratio=1/1, framerate={fps}/1 \
        ! decodebin \
        ! videoconvert \
        ! appsink"
    print(gst_str)

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise ValueError('Not Opened')

    caploop(cap)



if args.cap:
    gstreamer_rtp_cap()
else:
    gstreamer_rtp()