import os

input_imgs = 'inference/input'
output_dir = 'inference/output'
device = 'cpu'
iou = 0.3
conf = 0.5
img_size = 1600
weights_file = './weights/yolov5s_detect.pt'
# weights_file = 'C:/Users/amedhi/Documents/Dev/hklpr_yolov5/weights/yolov5s_detect.pt'

# Detect
# os.system(f"pwd")
os.system(f"python detect.py --img-size  {img_size} --conf-thres {conf}"
          f" --iou-thres {iou} --weights {weights_file} "
          f"--source {input_imgs} --device {device}")
# os.system(f"python detect.py --img-size  {img_size} --conf-thres {conf} --iou-thres {iou} --weights {weights_file} --source {input_imgs}")
