import json
import numpy as np
import os
import subprocess
# from azureml.core.model import Model

from lp_detect_recog import *

imgsz_detect = 1600
imgsz_recog = 224
device = 'cpu'


def execute(cmd):
    print(cmd)
    popen = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    return popen


def init():
    global model_detect, model_recog, imgsz_detect, imgsz_recog, names_detect, names_recog, half, img_lp, device

    try:    
        for line in execute(['ls', '-1']):
            print(line)
            for line in execute(['ls', line.strip()]):
                print(line)
    except Exception as e:
        print(e)

    # weights_detect = Model.get_model_path('lpr_detect')
    # weights_recog = Model.get_model_path('lpr_recog')
    weights_detect = './yolov5/weights/yolov5s_detect.pt'
    weights_recog = './yolov5/weights/yolov5s_recog.pt'
    # Load model
    model_detect = attempt_load(
        weights_detect, map_location=device)  # load FP32 model
    model_recog = attempt_load(
        weights_recog, map_location=device)  # load FP32 model
    imgsz_detect = check_img_size(
        imgsz_detect, s=model_detect.stride.max())  # check img_size
    imgsz_recog = check_img_size(
        imgsz_recog, s=model_recog.stride.max())  # check img_size

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    names_detect = model_detect.module.names if hasattr(
        model_detect, 'module') else model_detect.names
    names_recog = model_recog.module.names if hasattr(
        model_recog, 'module') else model_recog.names

    img = torch.zeros((1, 3, imgsz_detect, imgsz_detect),
                      device=device)  # init img
    img_lp = torch.zeros((1, 3, imgsz_recog, imgsz_recog),
                         device=device)  # init img
    if device.type != 'cpu':  # run once
        _ = model_detect(img.half() if half else img)
        _ = model_recog(img.half() if half else img)


def run(data):
    global img_lp
    try:
        im0s = np.array(json.loads(data))

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model_detect(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres_detect, opt.iou_thres_detect,
                                   classes=opt.classes_detect, agnostic=opt.agnostic_nms)

        # Process detections
        results = []
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s
            s = '%gx%g ' % img.shape[2:]  # print string

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += '%g %ss, ' % (n, names_detect[int(c)])

                # Write results
                # But first, Recognition
                img_lp0s = [extract_img_lp0(
                    im0, xyxy, img_lp, device, imgsz_recog, half) for *xyxy, _, _ in reversed(det)]
                img_lps = [extract_img_lp1(
                    img_lp0, img_lp, device, imgsz_recog, half) for img_lp0 in img_lp0s]

                height = max([x.shape[-2] for x in img_lps])
                width = max([x.shape[-1] for x in img_lps])
                img_lps = [torch.nn.ZeroPad2d(
                    (0, width-x.shape[-1], 0, height-x.shape[-2]))(x) for x in img_lps]
                img_lp = torch.cat(img_lps).to(device)

                t1 = time_synchronized()

                # Inference
                infs = model_recog(img_lp, augment=opt.augment)[0]

                # Apply NMS
                pred_lps = non_max_suppression(infs, opt.conf_thres_recog, opt.iou_thres_recog,
                                               classes=opt.classes_recog, agnostic=opt.agnostic_nms)

                # Write results
                for (*xyxy, conf, cls), img_lp, img_lp0, pred_lp in zip(reversed(det), img_lps, img_lp0s, pred_lps):
                    cls = check_lp_lines_type(pred_lp, cls, img_lp, img_lp0)

                    # Sort characters based on pred_lp
                    license_str = sort_characters(
                        pred_lp, cls, img_lp, img_lp0, names_recog)
                    if len(license_str) == 0:
                        continue

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls,
                                                                     *xywh)  # label format
                    s += license_str + ' ' + \
                        ('%g ' * len(line)).rstrip() % line + '\n'

            results.append(s)

        return results
    except Exception as e:
        error = str(e)
        return error
