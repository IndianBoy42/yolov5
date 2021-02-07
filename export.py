import os

img_size_detect = 1600
weights_file_detect = 'weights/yolov5s_detect.pt'

img_size_recog = 224
weights_file_recog = 'weights/yolov5s_recog.pt'

os.system(f"python models/export.py --weights {weights_file_detect} --img-size  {img_size_detect} --batch-size 1")
os.system(f"python models/export.py --weights {weights_file_recog} --img-size  {img_size_recog} --batch-size 1")
