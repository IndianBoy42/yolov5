import os

input_imgs = 'inference/input'
output_dir = 'inference/output'
device = 'cpu'

iou_detect = 0.3
conf_detect = 0.5
img_size_detect = 1600
weights_file_detect = 'weights/yolov5s_detect.pt'

iou_recog = 0.3
conf_recog = 0.5
img_size_recog = 224
weights_file_recog = 'weights/yolov5s_recog.pt'

# os.system(f"python models/export.py --weights {weights_file_detect} --img-size  {img_size_detect} --batch-size 1")
# os.system(f"python models/export.py --weights {weights_file_recog} --img-size  {img_size_recog} --batch-size 1")

# Detect and recog
os.system(f"python lp_detect_recog.py --img-size_detect  {img_size_detect} --img-size_recog  {img_size_recog}"
          f" --conf-thres_detect {conf_detect} --conf-thres_recog {conf_recog}"
          f" --iou-thres_detect {iou_recog} --iou-thres_detect {iou_recog}"
          f" --weights_detect {weights_file_detect} --weights_recog {weights_file_recog}"
          f" --source {input_imgs} --device={device}")
# os.system(f"python detect.py --img-size  {img_size} --conf-thres {conf} --iou-thres {iou} --weights {weights_file} --source {input_imgs} --output {output_dir} --device cpu")
