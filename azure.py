import json
import numpy as np
import os
import subprocess
from PIL import Image
import base64, json
# from azureml.core.model import Model

from lp_detect_recog import *

imgsz_detect = 1280
imgsz_recog = 224
device = 'cpu'

augment = False
conf_thres_detect = 0.5
iou_thres_detect = 0.3
classes_detect = None
conf_thres_recog = 0.5
iou_thres_recog = 0.5
classes_recog = None
agnostic_nms = False

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

def proc(img, im0s, view_img=False, save_img=False):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()

    # Inference
    pred = model_detect(img, augment=augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres_detect, iou_thres_detect,
                                classes=classes_detect, agnostic=agnostic_nms)

    t2 = time_synchronized()
    all_t2_t1 = t2-t1
    # Print time (inference + NMS)
    print('Done Detection. (%.3fs)' % (all_t2_t1))

    # Process detections
    results = []
    for i, det in enumerate(pred):  # detections per image
        im0 = im0s[i]
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
            infs = model_recog(img_lp, augment=augment)[0]

            # Apply NMS
            pred_lps = non_max_suppression(infs, conf_thres_recog, iou_thres_recog,
                                            classes=classes_recog, agnostic=agnostic_nms)

            t2 = time_synchronized()
            all_t2_t1 = all_t2_t1 + t2 - t1

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
                line = (cls, *xywh, conf) 

                s += license_str + ' ' + \
                    ('%g ' * len(line)).rstrip() % line + '\n'

                results.append({
                    'lp': license_str,
                    'cls': cls,
                    'x': xywh[0],
                    'y': xywh[1],
                    'w': xywh[2],
                    'h': xywh[3],
                    'conf': conf
                })

                if save_img or view_img:  # Add bbox to image
                    # label = '%s %.2f' % (names[int(cls)], conf)
                    label = '%s %.2f' % (license_str, conf)
                    line_thickness = 3 if im0.shape[0] < 500 else 4
                    plot_one_box(xyxy, im0, label=label,
                                color=colors[int(cls)], line_thickness=3)
        # Stream results
        if view_img:
            cv2.imshow('view', im0)
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
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

    # Print time (inference + NMS)
    print('%sDone. (%.3fs)' % (s, all_t2_t1))

    return results

def run(data):
    global img_lp
    try:
        # im0s = np.array(json.loads(data))
        b64png = json.loads(data)
        png = base64.b64decode(b64png)
        pilimage = Image.load(png)
        im0s = np.array(pilimage)

        # Padded resize
        img = letterbox(im0s, new_shape=imgsz_detect)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return proc(img, [im0s])
    except Exception as e:
        error = str(e)
        return error
