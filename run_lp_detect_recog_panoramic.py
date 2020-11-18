"""
Panoramic images are for demo only, illustrating that a single camera
with multiple lens can be capture lots of lp at the same time.
For deployment, use run_lp_detect_recog to process images with approximately
square size.

"""
import os
from glob import glob
import cv2
import shutil


def split_to_2images(imgs_dir, left_ratio=0.5):
    # adjust left_ratio in case the original image is split across a lp in the middle
    out_dir = os.path.join(imgs_dir, 'split')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)  # delete output folder
    os.makedirs(out_dir)

    file_list = sorted(glob(os.path.join(imgs_dir, '*.*')))
    # for im_dir in file_list:
    for count, im_dir in enumerate(file_list):
        img = cv2.imread(im_dir)  # BGR, HWC
        assert img is not None, 'Image Not Found ' + im_dir
        w1 = int(img.shape[1]*left_ratio)
        sub_img1 = img[:, :w1, :]
        sub_img2 = img[:, w1:, :]
        (dirpath, file_name) = os.path.split(im_dir)
        basename = os.path.splitext(file_name)[0]
        out_filename1 = dirpath + "/split/" + basename + "_sub1.jpg"
        out_filename2 = dirpath + "/split/" + basename + "_sub2.jpg"
        cv2.imwrite(out_filename1, sub_img1)
        cv2.imwrite(out_filename2, sub_img2)
    return out_dir


def merge2images(imgs_dir):
    file_list = sorted(glob(os.path.join(imgs_dir, '*.jpg')))
    file_list_odd = file_list[::2]
    file_list_even = file_list[1::2]
    assert len(file_list_odd) == len(file_list_even), 'Total number of sub-images should be even'
    # for im_dir in file_list:
    for count, im_dir in enumerate(file_list_odd):
        sub_img1 = cv2.imread(im_dir)  # BGR, HWC
        sub_img2 = cv2.imread(file_list_even[count])  # BGR, HWC
        img = cv2.hconcat([sub_img1, sub_img2])

        (dirpath, file_name) = os.path.split(im_dir)
        basename = os.path.splitext(file_name)[0][:-5]
        basedir = os.path.split(dirpath)[0]
        out_filename = basedir + "/" + basename + ".jpg"
        cv2.imwrite(out_filename, img)
    return


input_imgs = 'inference/input_panoramic'
output_dir = 'inference/output_panoramic'
device = '0,1'

iou_detect = 0.3
conf_detect = 0.5
img_size_detect = 1600
weights_file_detect = 'weights/yolov5s_detect.pt'

iou_recog = 0.3
conf_recog = 0.5
img_size_recog = 416
weights_file_recog = 'weights/yolov5s_recog.pt'

# split panoramic images into two sub-images
input_imgs_split = split_to_2images(input_imgs, left_ratio=0.5)
output_dir_split = os.path.join(output_dir, 'split')

# Detect and recog
os.system(f"python lp_detect_recog.py --img-size_detect  {img_size_detect} --img-size_recog  {img_size_recog}"
          f" --conf-thres_detect {conf_detect} --conf-thres_recog {conf_recog}"
          f" --iou-thres_detect {iou_recog} --iou-thres_detect {iou_recog}"
          f" --weights_detect '{weights_file_detect}' --weights_recog '{weights_file_recog}'"
          f" --source '{input_imgs_split}' --output '{output_dir_split }' --device={device}")

# merge two sub-images into one
merge2images(output_dir_split)

# delete cache files and images
shutil.rmtree(input_imgs_split)  # delete output folder
shutil.rmtree(output_dir_split)  # delete output folder
