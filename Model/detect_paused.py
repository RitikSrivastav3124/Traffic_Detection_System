import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from types import SimpleNamespace

# =====================================
# PATHS
# =====================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (
    check_img_size,
    non_max_suppression,
    print_args,
    scale_boxes
)
from utils.torch_utils import select_device, smart_inference_mode

# =====================================
# BYTETRACK IMPORT
# =====================================
FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]          # Model
PROJECT_ROOT = FILE.parents[1] # Traffic_Detection

BYTE_PATH = PROJECT_ROOT / "ByteTrack"

sys.path.append(str(BYTE_PATH))

from yolox.tracker.byte_tracker import BYTETracker

# =====================================
# SETTINGS
# =====================================
NUM_LANES = 4

INITIAL_SCAN_TIME = 1      # start me ek baar all lanes
MIN_GREEN_TIME = 15        # seconds
MAX_WAIT_TIME = 60         # seconds
FPS = 30

current_green = 0
mode = "INITIAL_SCAN"
mode_timer = 0

lane_wait_time = [0] * NUM_LANES
last_lane_counts = [0] * NUM_LANES

# =====================================
# BYTE TRACKER CONFIG
# =====================================
tracker_args = SimpleNamespace(
    track_thresh=0.25,
    track_buffer=60,
    match_thresh=0.8,
    mot20=False
)

trackers = [BYTETracker(tracker_args, frame_rate=FPS) for _ in range(NUM_LANES)]


# =====================================
# DETECTION + TRACKING
# =====================================
def detect_and_track(frame, lane_id, model, device,
                     imgsz, stride, names,
                     conf_thres, iou_thres):

    im0 = frame.copy()

    img = letterbox(im0, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    detections = []
    emergency = False

    for det in pred:

        if len(det):

            det[:, :4] = scale_boxes(
                img.shape[2:],
                det[:, :4],
                im0.shape
            ).round()

            for *xyxy, conf, cls in det:

                x1, y1, x2, y2 = map(float, xyxy)
                score = float(conf)
                label = names[int(cls)].lower()

                detections.append(
                    [x1, y1, x2, y2, score]
                )

                if label == "ambulance":
                    emergency = True

    # ByteTrack update
    if len(detections) > 0:
        detections = np.array(detections, dtype=np.float32)
    else:
        detections = np.empty((0, 5), dtype=np.float32)

    online_targets = trackers[lane_id].update(
        detections,
        [im0.shape[0], im0.shape[1]],
        [im0.shape[0], im0.shape[1]]
    )

    count = 0

    for t in online_targets:

        tlwh = t.tlwh
        track_id = t.track_id

        x1, y1, w, h = tlwh
        x2 = x1 + w
        y2 = y1 + h

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        count += 1

        cv2.rectangle(
            im0,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            im0,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    return im0, count, emergency


# =====================================
# PRIORITY
# =====================================
def get_best_lane():

    global lane_wait_time, last_lane_counts

    # max waiting lane first
    for i in range(NUM_LANES):
        if lane_wait_time[i] >= MAX_WAIT_TIME * FPS:
            return i

    # else max vehicle lane
    return int(np.argmax(last_lane_counts))


# =====================================
# MAIN
# =====================================
@smart_inference_mode()
def run(weights,
        sources,
        imgsz=(640, 640),
        conf_thres=0.35,
        iou_thres=0.45,
        device="",
        view_img=True):

    global current_green, mode, mode_timer
    global lane_wait_time, last_lane_counts

    device = select_device(device)

    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    imgsz = check_img_size(imgsz, s=stride)

    caps = [cv2.VideoCapture(src) for src in sources]

    os.makedirs("output", exist_ok=True)

    writers = []
    paused_frames = []

    for i, cap in enumerate(caps):

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        writers.append(
            cv2.VideoWriter(
                f"output/lane{i+1}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )
        )

        paused_frames.append(
            np.zeros((h, w, 3), dtype=np.uint8)
        )

    while True:

        frames = []
        emergency_lane = -1

        # =====================================
        # INITIAL SCAN
        # =====================================
        if mode == "INITIAL_SCAN":

            for i, cap in enumerate(caps):

                ret, frame = cap.read()

                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                processed, count, emergency = detect_and_track(
                    frame, i, model, device,
                    imgsz, stride, names,
                    conf_thres, iou_thres
                )

                if emergency:
                    emergency_lane = i

                last_lane_counts[i] = count
                paused_frames[i] = processed.copy()
                frames.append(processed)

            mode_timer += 1

            if mode_timer >= INITIAL_SCAN_TIME * FPS:

                if emergency_lane != -1:
                    current_green = emergency_lane
                else:
                    current_green = get_best_lane()

                mode = "GREEN"
                mode_timer = 0

        # =====================================
        # GREEN MODE
        # =====================================
        else:

            for i, cap in enumerate(caps):

                if i != current_green:
                    frames.append(paused_frames[i].copy())
                    continue

                ret, frame = cap.read()

                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                processed, count, emergency = detect_and_track(
                    frame, i, model, device,
                    imgsz, stride, names,
                    conf_thres, iou_thres
                )

                if emergency:
                    emergency_lane = i

                last_lane_counts[i] = count
                paused_frames[i] = processed.copy()
                frames.append(processed)

            mode_timer += 1

            # after green time snapshot all
            if mode_timer >= MIN_GREEN_TIME * FPS:

                for i, cap in enumerate(caps):

                    ret, frame = cap.read()

                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()

                    processed, count, emergency = detect_and_track(
                        frame, i, model, device,
                        imgsz, stride, names,
                        conf_thres, iou_thres
                    )

                    if emergency:
                        emergency_lane = i

                    last_lane_counts[i] = count
                    paused_frames[i] = processed.copy()

                if emergency_lane != -1:
                    current_green = emergency_lane
                else:
                    current_green = get_best_lane()

                mode_timer = 0

        # =====================================
        # WAIT TIME
        # =====================================
        for i in range(NUM_LANES):

            if i == current_green:
                lane_wait_time[i] = 0
            else:
                lane_wait_time[i] += 1

                if lane_wait_time[i] > MAX_WAIT_TIME * FPS:
                    lane_wait_time[i] = MAX_WAIT_TIME * FPS

        # =====================================
        # DISPLAY
        # =====================================
        signals = ["RED"] * NUM_LANES
        signals[current_green] = "GREEN"

        for i, frame in enumerate(frames):

            color = (0, 255, 0) if signals[i] == "GREEN" else (0, 0, 255)

            cv2.putText(
                frame,
                f"Lane {i+1}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Vehicles: {last_lane_counts[i]}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Wait: {lane_wait_time[i]//FPS}s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Signal: {signals[i]}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

            if view_img:
                cv2.imshow(f"Camera {i+1}", frame)

            writers[i].write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    for cap in caps:
        cap.release()

    for w in writers:
        w.release()

    cv2.destroyAllWindows()


# =====================================
# ARGUMENTS
# =====================================
def parse_opt():

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", nargs='+', type=str, required=True)
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument("--view-img", action="store_true")

    opt = parser.parse_args()

    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    print_args(vars(opt))
    return opt


def main(opt):

    run(
        opt.weights,
        opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        view_img=opt.view_img
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)