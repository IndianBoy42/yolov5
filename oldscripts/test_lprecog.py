import os

# weights_file = 'weights/lp_recognition.pt'
# weights_file = 'runs/exp17_yolov5s_lprecog_results/weights/best.pt'
weights_file = 'runs/exp19_yolov5s_lprecog_results/weights/best.pt'
# weights_file = 'runs/exp5_yolov5s_lprecog_results/weights/best.pt'
dataset_yaml = 'data/lp_recognition.yaml'
# n_gpu = 1
# device = '1'
n_gpu = 2
device = '0,1'
batch_size_per_gpu = 256
batch_size = n_gpu * batch_size_per_gpu
img_size_test = 416
iou = 0.3
conf = 0.6
model_yaml = 'yolov5s_lprecog'
task = 'test'


# Testing
# os.system(f"python test.py --img-size {img_size_test} --batch-size {batch_size}"
#           f" --data '{dataset_yaml}' --weights '{weights_file}'"
#           f" --conf-thres {conf} --iou-thres {iou} --task '{task}' --verbose")
# with Merge NMS better performance
os.system(f"python test.py --img-size {img_size_test} --batch-size {batch_size}"
          f" --data '{dataset_yaml}' --weights '{weights_file}' --device={device}"
          f" --conf-thres {conf} --iou-thres {iou} --task '{task}' --merge --verbose")
