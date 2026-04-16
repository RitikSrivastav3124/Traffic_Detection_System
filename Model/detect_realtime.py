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

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

# -------- SIGNAL SYSTEM --------
lane_wait_time = [0, 0, 0, 0]
green_timer = 0
current_green = -1

MIN_GREEN_TIME = 2
MAX_GREEN_TIME = 5
FPS = 30

# -------- PERFORMANCE --------
FRAME_SKIP = 2


@smart_inference_mode()
def run(weights, sources, conf_thres=0.25, device="", view_img=True):

    global lane_wait_time, green_timer, current_green

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    names = model.names

    # -------- LOAD CAMERAS --------
    caps = []
    for src in sources:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"❌ Cannot open {src}")
        else:
            print(f"✅ Camera connected: {src}")

        caps.append(cap)

    frame_count = 0

    while True:

        frames = []
        lane_counts = []
        emergency_lane = -1

        frame_count += 1

        for cam_id, cap in enumerate(caps):

            ret, frame = cap.read()

            # -------- RECONNECT --------
            if not ret:
                print(f"⚠️ Reconnecting Camera {cam_id}")
                cap.release()
                cap = cv2.VideoCapture(sources[cam_id])
                caps[cam_id] = cap
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                lane_counts.append(0)
                continue

            # -------- FRAME SKIP --------
            if frame_count % FRAME_SKIP != 0:
                frames.append(frame)
                lane_counts.append(0)
                continue

            im0 = frame.copy()

            # -------- PREPROCESS --------
            img = cv2.resize(im0, (640, 640))
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float() / 255.0
            img = img.unsqueeze(0)

            # -------- INFERENCE --------
            pred = model(img)
            pred = non_max_suppression(pred, conf_thres, 0.45)

            count = 0

            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape)

                    for *xyxy, conf, cls in det:
                        label = names[int(cls)].lower()

                        # only vehicles
                        if label not in ["car", "truck", "bus", "motorbike", "bicycle"]:
                            continue

                        x1, y1, x2, y2 = map(int, xyxy)
                        count += 1

                        # emergency detection
                        if "ambulance" in label or "fire" in label or "police" in label:
                            emergency_lane = cam_id

                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(im0, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            lane_counts.append(count)
            frames.append(im0)

        # -------- SIGNAL LOGIC --------
        priority_scores = [
            lane_counts[i] * 3 + lane_wait_time[i]
            for i in range(len(lane_counts))
        ]

        if emergency_lane != -1:
            current_green = emergency_lane
            green_timer = 0
        else:
            if current_green == -1:
                current_green = int(np.argmax(priority_scores))
                green_timer = 0
            else:
                green_timer += 1

                if green_timer > MIN_GREEN_TIME * FPS:
                    best_lane = int(np.argmax(priority_scores))

                    if best_lane != current_green or green_timer > MAX_GREEN_TIME * FPS:
                        current_green = best_lane
                        green_timer = 0

        green_lane = current_green

        # -------- WAIT UPDATE --------
        for i in range(len(lane_wait_time)):
            if i == green_lane:
                lane_wait_time[i] = 0
            else:
                lane_wait_time[i] += 1

        signals = ["RED"] * len(frames)
        signals[green_lane] = "GREEN"

        # -------- DISPLAY --------
        for i, frame in enumerate(frames):

            color = (0, 255, 0) if signals[i] == "GREEN" else (0, 0, 255)

            cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(frame, f"Signal: {signals[i]}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.putText(frame, f"Wait: {lane_wait_time[i]//FPS}s", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if view_img:
                cv2.imshow(f"Camera {i+1}", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", nargs='+', type=str, required=True)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--view-img", action="store_true")

    return parser.parse_args()


def main(opt):
    run(opt.weights, opt.source,
        conf_thres=opt.conf_thres,
        view_img=opt.view_img)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)