#!/usr/bin/env python3

import io
import time
import picamera
from PIL import Image
import base64, json

def iter(start=0):
    while True:
        yield start
        start += 1

# Create the in-memory stream
outdir = '../imgs'
with picamera.PiCamera() as camera:
    # camera.start_preview()
    # time.sleep(2)
    print("Start")
    for i in iter():
        stream = io.BytesIO()

        camera.capture(stream, format='png')
        # "Rewind" the stream to the beginning so we can read its content
        print("Captured")
        stream.seek(0)

        output = base64.b64encode(stream.read()).decode('utf-8')
        print(type(output))
        json.dump({"data": output}, open(f'{outdir}/test{i}.json', 'w'))
        req = json.dumps({"data": output})

        # image = Image.open(stream)
        # print(image)
        # image.save(f'{outdir}/cap{i}.png')

        # time.sleep(1)
        print("Loop")
        input("Next frame")
        
