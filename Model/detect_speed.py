import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    check_img_size,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode

# ByteTrack
from yolox.tracker.byte_tracker import BYTETracker

heatmap = None

# ---------- SPEED ----------
vehicle_positions = {}
vehicle_speeds = {}

pixels_per_meter = 25


@smart_inference_mode()
def run(
        weights,
        source,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        device="",
        view_img=False,
        project=ROOT / "runs/detect",
        name="exp",
        exist_ok=False
):

    global heatmap

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)

    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    vid_path, vid_writer = None, None

    model.warmup(imgsz=(1, 3, *imgsz))

    # -------- ByteTrack Tracker --------
    tracker = BYTETracker(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30
    )

    for frame_id, (path, im, im0s, vid_cap, s) in enumerate(dataset):

        start = time.time()

        im = torch.from_numpy(im).to(model.device)
        im = im.float() / 255.0

        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred:

            im0 = im0s.copy()
            H, W, _ = im0.shape

            if heatmap is None:
                heatmap = np.zeros((H, W), dtype=np.float32)

            detections = []

            if len(det):

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape)

                for *xyxy, conf, cls in det:

                    x1,y1,x2,y2 = map(int,xyxy)

                    detections.append(
                        [x1,y1,x2,y2,conf.item()]
                    )

            detections = np.array(detections)

            # -------- TRACKING --------
            tracks = tracker.update(detections,(H,W),(H,W))

            vehicle_total = len(tracks)

            for track in tracks:

                x1,y1,x2,y2 = map(int,track.tlbr)

                track_id = int(track.track_id)

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                heatmap[cy,cx]+=1

                label="vehicle"

                # -------- SPEED --------
                if track_id in vehicle_positions:

                    prev_x,prev_y=vehicle_positions[track_id]

                    pixel_distance=np.sqrt(
                        (cx-prev_x)**2+(cy-prev_y)**2
                    )

                    distance_m=pixel_distance/pixels_per_meter

                    time_sec=1/30

                    speed=(distance_m/time_sec)*3.6

                    vehicle_speeds[track_id]=int(speed)

                else:
                    vehicle_speeds[track_id]=0

                vehicle_positions[track_id]=(cx,cy)

                speed=vehicle_speeds.get(track_id,0)

                # -------- DRAW --------
                cv2.rectangle(im0,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(im0,
                            f"{label} ID:{track_id} {speed} km/h",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,255),
                            2)

                cv2.circle(im0,(cx,cy),4,(0,255,255),-1)

            fps=int(1/(time.time()-start))

            cv2.putText(im0,f"FPS {fps}",
                        (W-150,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),2)

            # -------- HEATMAP --------
            heatmap_norm=cv2.normalize(
                heatmap,None,0,255,cv2.NORM_MINMAX
            )

            heatmap_color=cv2.applyColorMap(
                heatmap_norm.astype(np.uint8),
                cv2.COLORMAP_JET
            )

            im0=cv2.addWeighted(im0,0.85,heatmap_color,0.15,0)

            if view_img:
                cv2.imshow("Traffic Detection",im0)
                cv2.waitKey(1)

            save_path=str(save_dir/Path(path).name)

            if vid_path!=save_path:

                vid_path=save_path

                if isinstance(vid_writer,cv2.VideoWriter):
                    vid_writer.release()

                if vid_cap:
                    fps=vid_cap.get(cv2.CAP_PROP_FPS)
                    w=int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h=int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    fps=30
                    h,w=im0.shape[:2]

                save_path=str(Path(save_path).with_suffix(".mp4"))

                vid_writer=cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w,h)
                )

            vid_writer.write(im0)

    LOGGER.info(f"Results saved to {save_dir}")


def parse_opt():

    parser=argparse.ArgumentParser()

    parser.add_argument("--weights",type=str)
    parser.add_argument("--source",type=str)
    parser.add_argument("--imgsz",nargs="+",type=int,default=[640])
    parser.add_argument("--conf-thres",type=float,default=0.25)
    parser.add_argument("--view-img",action="store_true")

    opt=parser.parse_args()

    opt.imgsz*=2 if len(opt.imgsz)==1 else 1

    print_args(vars(opt))

    return opt


def main(opt):
    run(**vars(opt))


if __name__=="__main__":

    opt=parse_opt()
    main(opt)