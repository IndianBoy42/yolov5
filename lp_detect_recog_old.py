import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from utils.datasets import letterbox
import numpy as np


# TODO:
def check_lp_lines_type(det, lp_lines_type, img_lp, img_lp0):
    gn = torch.tensor(img_lp0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if lp_lines_type == 0:
        """                      
        if actually two lines lp
        |   A B   |  
        | 1 2 3 4 |
        , then averaged y of '1' '4' should be quite different from averaged y of other characters or digits  
        """
        if det is not None and len(det)>2:
            xywh_list = []
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_lp.shape[2:], det[:, :4], img_lp0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                xywh_list.append(xywh)
            sorted_xywh_list = [x for x in sorted(xywh_list, key=lambda xywh_list: xywh_list[0])]
            ave_h = np.average(sorted_xywh_list, 0)[3]
            ave_y_outer = (sorted_xywh_list[0][1] + sorted_xywh_list[-1][1])/2
            ave_y_inner = np.average(sorted_xywh_list[1:-1], 0)[1]
            if abs(ave_y_outer - ave_y_inner) >  ave_h/3:
                lp_lines_type = lp_lines_type+1

    return lp_lines_type


def sort_characters(det, lp_lines_type, img_lp, img_lp0, names_recog):
    gn = torch.tensor(img_lp0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    line1_xywhc_list = []
    line2_xywhc_list = []
    sorted_line1_xywhc_list = []
    sorted_line2_xywhc_list = []
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img_lp.shape[2:], det[:, :4], img_lp0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            cls = int(cls.data.tolist())
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            # xywh.append(cls)
            if lp_lines_type == 0 or cls > 9:
                line1_xywhc_list.append(xywh+[cls])
            else:
                line2_xywhc_list.append(xywh+[cls])
        sorted_line1_xywhc_list = [x for x in sorted(line1_xywhc_list, key=lambda line1_xywhc_list: line1_xywhc_list[0])]
        if len(line2_xywhc_list) > 0:
            sorted_line2_xywhc_list = [x for x in sorted(line2_xywhc_list, key=lambda line2_xywhc_list: line2_xywhc_list[0])]
    line1_license_str = ''.join([names_recog[xywhc[4]] for xywhc in sorted_line1_xywhc_list])
    line2_license_str = ''.join([names_recog[xywhc[4]] for xywhc in sorted_line2_xywhc_list])
    return line1_license_str + line2_license_str


def detect_recog(save_img=False):
    out, source, weights_detect, weights_recog, view_img, save_txt, imgsz_detect, imgsz_recog = \
        opt.output, opt.source, opt.weights_detect, opt.weights_recog, opt.view_img, opt.save_txt,\
        opt.img_size_detect, opt.img_size_recog
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_detect = attempt_load(weights_detect, map_location=device)  # load FP32 model
    model_recog = attempt_load(weights_recog, map_location=device)  # load FP32 model
    imgsz_detect = check_img_size(imgsz_detect, s=model_detect.stride.max())  # check img_size
    imgsz_recog = check_img_size(imgsz_recog, s=model_recog.stride.max())  # check img_size
    if half:
        model_detect.half()  # to FP16
        model_recog.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz_detect)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz_detect)

    # Get names and colors
    names_detect = model_detect.module.names if hasattr(model_detect, 'module') else model_detect.names
    names_recog = model_detect.module.names if hasattr(model_recog, 'module') else model_recog.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names_detect))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz_detect, imgsz_detect), device=device)  # init img
    img_lp = torch.zeros((1, 3, imgsz_recog, imgsz_recog), device=device)  # init img
    _ = model_detect(img.half() if half else img) if device.type != 'cpu' else None  # run once
    _ = model_recog(img_lp.half() if half else img_lp) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model_detect(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres_detect, opt.iou_thres_detect, classes=opt.classes_detect, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        all_t2_t1 = t2-t1

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names_detect[int(c)])  # add to string

                # Recognition and then Write results
                for *xyxy, conf, cls in reversed(det):

                    ''' Recognition '''
                    # Retrieve original resolution for each lp
                    img_lp0 = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]   # BGR
                    # Padded resize
                    img_lp = letterbox(img_lp0, new_shape=imgsz_recog)[0]
                    # Convert
                    img_lp = img_lp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img_lp = np.ascontiguousarray(img_lp)
                    img_lp = torch.from_numpy(img_lp).to(device)
                    img_lp = img_lp.half() if half else img_lp.float()  # uint8 to fp16/32
                    img_lp /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img_lp.ndimension() == 3:
                        img_lp = img_lp.unsqueeze(0)
                    t1 = time_synchronized()
                    # Inference
                    pred_lp = model_recog(img_lp, augment=opt.augment)[0]
                    # Apply NMS
                    pred_lp = non_max_suppression(pred_lp, opt.conf_thres_recog, opt.iou_thres_recog,
                                                  classes=opt.classes_recog, agnostic=opt.agnostic_nms)
                    t2 = time_synchronized()
                    all_t2_t1 = all_t2_t1 + t2 - t1
                    # Apply Classifier
                    if classify:
                        pred_lp = apply_classifier(pred_lp, modelc, img_lp, img_lp0)
                    # check_lp_lines_type
                    cls = check_lp_lines_type(pred_lp[0], cls, img_lp, img_lp0)
                    # Sort characters based on pred_lp
                    license_str = sort_characters(pred_lp[0], cls, img_lp, img_lp0, names_recog)
                    if len(license_str) == 0:
                        continue

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s %.2f' % (license_str, conf)
                        line_thickness = 3 if im0.shape[0] < 500 else 4
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=line_thickness)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, all_t2_t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_detect', nargs='+', type=str, default='weights/yolov5s_detect.pt', help='model.pt path(s)')
    parser.add_argument('--weights_recog', nargs='+', type=str, default='weights/yolov5s_recog.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/input', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size_detect', type=int, default=1600, help='inference size (pixels)')
    parser.add_argument('--img-size_recog', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres_detect', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--conf-thres_recog', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres_detect', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--iou-thres_recog', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes_detect', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--classes_recog', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
        #         detect_recog()
        #         strip_optimizer(opt.weights)
        # else:
        detect_recog()