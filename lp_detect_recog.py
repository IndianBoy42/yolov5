import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.autograd.profiler as profiler
from numpy import random

from threading import Thread

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
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
        if det is not None and len(det) > 2:
            xywh_list = []
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img_lp.shape[2:], det[:, :4], img_lp0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                        gn).view(-1).tolist()  # normalized xywh
                xywh_list.append(xywh)
            sorted_xywh_list = [x for x in sorted(
                xywh_list, key=lambda xywh_list: xywh_list[0])]
            ave_h = np.average(sorted_xywh_list, 0)[3]
            ave_y_outer = (sorted_xywh_list[0][1] + sorted_xywh_list[-1][1])/2
            ave_y_inner = np.average(sorted_xywh_list[1:-1], 0)[1]
            if abs(ave_y_outer - ave_y_inner) > ave_h/3:
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
        det[:, :4] = scale_coords(
            img_lp.shape[2:], det[:, :4], img_lp0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            cls = int(cls.data.tolist())
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                    gn).view(-1).tolist()  # normalized xywh
            # xywh.append(cls)
            if lp_lines_type == 0 or cls > 9:
                line1_xywhc_list.append(xywh+[cls])
            else:
                line2_xywhc_list.append(xywh+[cls])
        sorted_line1_xywhc_list = [x for x in sorted(
            line1_xywhc_list, key=lambda line1_xywhc_list: line1_xywhc_list[0])]
        if len(line2_xywhc_list) > 0:
            sorted_line2_xywhc_list = [x for x in sorted(
                line2_xywhc_list, key=lambda line2_xywhc_list: line2_xywhc_list[0])]
    line1_license_str = ''.join([names_recog[xywhc[4]]
                                 for xywhc in sorted_line1_xywhc_list])
    line2_license_str = ''.join([names_recog[xywhc[4]]
                                 for xywhc in sorted_line2_xywhc_list])
    return line1_license_str + line2_license_str


def extract_img_lp0(im0, xyxy, img_lp, device, imgsz_recog, half):
    # Retrieve original resolution for each lp
    img_lp0 = im0[int(xyxy[1]):int(xyxy[3]), int(
        xyxy[0]):int(xyxy[2]), :]   # BGR

    return img_lp0


def extract_img_lp1(img_lp0, img_lp, device, imgsz_recog, half):
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

    return img_lp


def extract_img_lp(im0, xyxy, img_lp, device, imgsz_recog, half):
    img_lp0 = extract_img_lp0(im0, xyxy, img_lp, device, imgsz_recog, half)
    img_lp = extract_img_lp1(img_lp0, img_lp, device, imgsz_recog, half)
    return img_lp, img_lp0


def recog2(det, im0, device, img_lp, imgsz_recog, half, model_recog, all_t2_t1, classify, modelc, names_recog, save_txt, gn, txt_path, save_img, view_img, colors):
    # Write results
    for *xyxy, conf, cls in reversed(det):
        ''' But first, Recognition '''
        img_lp, img_lp0 = extract_img_lp(
            im0, xyxy, img_lp, device, imgsz_recog, half)

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
        license_str = sort_characters(
            pred_lp[0], cls, img_lp, img_lp0, names_recog)
        if len(license_str) == 0:
            continue

        if save_txt:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                    gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if opt.save_conf else (cls,
                                                             *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(license_str + ' ' +
                        ('%g ' * len(line)).rstrip() % line + '\n')

        if save_img or view_img:  # Add bbox to image
            # label = '%s %.2f' % (names[int(cls)], conf)
            label = '%s %.2f' % (license_str, conf)
            line_thickness = 3 if im0.shape[0] < 500 else 4
            plot_one_box(xyxy, im0, label=label,
                         color=colors[int(cls)], line_thickness=3)

    return all_t2_t1


def recog(det, im0, device, img_lp, imgsz_recog, half, model_recog, all_t2_t1, classify, modelc, names_recog, save_txt, gn, txt_path, save_img, view_img, colors):
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

    t2 = time_synchronized()
    all_t2_t1 = all_t2_t1 + t2 - t1

    # Write results
    for (*xyxy, conf, cls), img_lp, img_lp0, pred_lp in zip(reversed(det), img_lps, img_lp0s, pred_lps):
        # Apply Classifier
        if classify:
            pred_lp = apply_classifier(pred_lp, modelc, img_lp, img_lp0)

        cls = check_lp_lines_type(pred_lp, cls, img_lp, img_lp0)

        # Sort characters based on pred_lp
        license_str = sort_characters(
            pred_lp, cls, img_lp, img_lp0, names_recog)
        if len(license_str) == 0:
            continue

        if save_txt:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                    gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if opt.save_conf else (cls,
                                                             *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(license_str + ' ' +
                        ('%g ' * len(line)).rstrip() % line + '\n')

        if save_img or view_img:  # Add bbox to image
            # label = '%s %.2f' % (names[int(cls)], conf)
            label = '%s %.2f' % (license_str, conf)
            line_thickness = 3 if im0.shape[0] < 500 else 4
            plot_one_box(xyxy, im0, label=label,
                         color=colors[int(cls)], line_thickness=3)

    return all_t2_t1


def detect_recog():
    source, weights_detect, weights_recog, view_img, save_txt, imgsz_detect, imgsz_recog, save_img = opt.source, opt.weights_detect, opt.weights_recog, opt.view_img, opt.save_txt, opt.img_size_detect, opt.img_size_recog, opt.save_img

    # Set Dataloader
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    vid_path, vid_writer = None, None
    shmstream = source.startswith('/tmp/')
    if shmstream:
        source = f"shmsrc socket-path={source} \
                ! video/x-raw, format=BGR, width={int(imgsz_detect*4/3)}, height={imgsz_detect}, pixel-aspect-ratio=1/1, framerate=30/1 \
                ! decodebin \
                ! videoconvert \
                ! appsink"
        dataset = LoadStreamsBuffered(source, img_size=imgsz_detect)
    elif webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz_detect)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz_detect)

    # Directories
    if opt.save_dir == 'runs/exp':
        save_dir = Path(increment_path(Path(opt.project) /
                                       opt.name, exist_ok=opt.exist_ok))  # increment run
    else:
        save_dir = Path(opt.save_dir)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_detect = attempt_load(
        weights_detect, map_location=device)  # load FP32 model
    model_recog = attempt_load(
        weights_recog, map_location=device)  # load FP32 model
    imgsz_detect = check_img_size(
        imgsz_detect, s=model_detect.stride.max())  # check img_size
    imgsz_recog = check_img_size(
        imgsz_recog, s=model_recog.stride.max())  # check img_size
    if half:
        model_detect.half()  # to FP16
        model_recog.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    else:
        modelc = None

    # Get names and colors
    names_detect = model_detect.module.names if hasattr(
        model_detect, 'module') else model_detect.names
    names_recog = model_recog.module.names if hasattr(
        model_recog, 'module') else model_recog.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names_detect]

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz_detect, imgsz_detect),
                      device=device)  # init img
    img_lp = torch.zeros((1, 3, imgsz_recog, imgsz_recog),
                         device=device)  # init img
    if device.type != 'cpu':  # run once
        _ = model_detect(img.half() if half else img)
        _ = model_recog(img.half() if half else img)

    # Run inference
    shmcounter = 0
    for path, img, im0s, vid_cap in dataset:
        if img is None:
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()

        # Inference
        pred = model_detect(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres_detect, opt.iou_thres_detect,
                                   classes=opt.classes_detect, agnostic=opt.agnostic_nms)

        t2 = time_synchronized()
        all_t2_t1 = t2-t1

        # Print time (inference + NMS)
        print('Done Detection. (%.3fs)' % (all_t2_t1))

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if shmstream:
                p, s, im0 = Path(
                    f"{shmcounter}.jpg"), '%g: ' % i, im0s[i].copy()
                shmcounter += 1
            elif webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' %
                                                            dataset.frame if dataset.mode == 'video' else '')

            s += '%gx%g ' % img.shape[2:]  # print string
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
                all_t2_t1 = recog(det, im0, device, img_lp, imgsz_recog, half, model_recog, all_t2_t1,
                                  classify, modelc, names_recog, save_txt, gn, txt_path, save_img, view_img, colors)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
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
        print('%sDone Recognition. (%.3fs)' % (s, all_t2_t1))

    if save_txt or save_img:
        print('Results saved to %s' % save_dir)

    print('Done. (%.3fs)' % (time.time() - t0))


class LoadStreamsBuffered:  # multiple IP or RTSP cameras
    def __init__(self, source='/tmp/shm', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        self.imgs = []
        self.source = source

        # Start the thread to read frames from the video stream
        print('%g/%g: %s... ' % (1, 1, source), end='')
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), 'Failed to open %s' % source

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) % 100

        _, tmp = cap.read()  # guarantee first frame
        self.imgs.append(tmp)

        thread = Thread(target=self.update, args=([cap]), daemon=True)
        print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
        thread.start()

        # check for common shapes
        self.rect = True

    def update(self, cap):
        # Read next stream frame in a daemon thread
        while cap.isOpened():
            # _, self.imgs[index][self.heads[index]] = cap.read()
            cap.grab()

            _, tmp = cap.retrieve()
            self.imgs.append(tmp)

            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        # FIXME: Race Conditions??
        img0 = self.imgs
        self.imgs = []
        # TODO: block if no new images

        if len(img0) == 0:
            return self.source, None, img0, None
        else:
            # if cv2.waitKey(1) == ord('q'):  # q to quit
            # cv2.destroyAllWindows()
            # raise StopIteration

            # Letterbox
            img = [letterbox(x, new_shape=self.img_size,
                             auto=self.rect)[0] for x in img0]

            # Stack
            img = np.stack(img, 0)

            # Convert
            # BGR to RGB, to bsx3x416x416
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
            img = np.ascontiguousarray(img)

            print(f'Recieved {img.shape}')

            return self.source, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_detect', nargs='+', type=str,
                        default='weights/yolov5s_detect.pt', help='model.pt path(s)')
    parser.add_argument('--weights_recog', nargs='+', type=str,
                        default='weights/yolov5s_recog.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/input', help='source')
    parser.add_argument('--img-size_detect', type=int,
                        default=1600, help='inference size (pixels)')
    parser.add_argument('--img-size_recog', type=int,
                        default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres_detect', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--conf-thres_recog', type=float,
                        default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres_detect', type=float,
                        default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--iou-thres_recog', type=float,
                        default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-dir', type=str,
                        default='runs/exp', help='save directory')
    parser.add_argument('--save-img', action='store_true', help='save images')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes_detect', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--classes_recog', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect_recog()
    # with profiler.profile(profile_memory=True) as prof:
        # with torch.no_grad():
        #     detect_recog()
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")
