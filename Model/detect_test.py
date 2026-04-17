import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, print_args, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox  # ✅ FIX

# -------- TRACKING --------
vehicle_tracks = {}
vehicle_speeds = {}
next_vehicle_id = 0

pixels_per_meter = 155
max_tracking_distance = 50

# -------- SIGNAL SYSTEM --------
lane_wait_time = [0, 0, 0, 0]
green_timer = 0
current_green = -1

MIN_GREEN_TIME = 2
MAX_GREEN_TIME = 4
FPS = 30


@smart_inference_mode()
def run(weights, sources, imgsz=(640, 640), conf_thres=0.3, iou_thres=0.45, device="", view_img=True):

    global next_vehicle_id, lane_wait_time, green_timer, current_green

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    imgsz = check_img_size(imgsz, s=stride)

    caps = []
    finished = [False] * len(sources)

    for src in sources:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"❌ Cannot open {src}")
        else:
            print(f"✅ Loaded {src}")
        caps.append(cap)

    REAL_FPS = caps[0].get(cv2.CAP_PROP_FPS)
    if REAL_FPS == 0 or REAL_FPS is None:
        REAL_FPS = 30

    print(f"🎯 Using FPS: {REAL_FPS}")

    while True:

        frames = []
        lane_counts = []
        all_finished = True
        emergency_lane = -1

        # -------- DETECTION LOOP --------
        for cam_id, cap in enumerate(caps):

            if finished[cam_id]:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                lane_counts.append(0)
                continue

            ret, frame = cap.read()

            if not ret:
                finished[cam_id] = True
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                lane_counts.append(0)
                continue
            else:
                all_finished = False

            im0 = frame.copy()

            # ✅ LETTERBOX FIX (IMPORTANT)
            img = letterbox(im0, imgsz, stride=stride, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float() / 255.0
            img = img.unsqueeze(0)

            pred = model(img)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            count = 0

            for det in pred:
                if len(det):

                    # ✅ SCALE FIX
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det:

                        x1, y1, x2, y2 = map(int, xyxy)

                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        current_center = np.array([cx, cy])
                        matched_id = None

                        # -------- TRACKING --------
                        for vid, prev_center in vehicle_tracks.items():
                            if np.linalg.norm(current_center - prev_center) < max_tracking_distance:
                                matched_id = vid
                                break

                        if matched_id is None:
                            matched_id = next_vehicle_id
                            next_vehicle_id += 1
                            vehicle_speeds[matched_id] = 0

                        if matched_id in vehicle_tracks:
                            prev_center = vehicle_tracks[matched_id]
                            pixel_distance = np.linalg.norm(current_center - prev_center)
                            distance_m = pixel_distance / pixels_per_meter

                            raw_speed = distance_m * REAL_FPS * 3.6
                            alpha = 0.8
                            prev_speed = vehicle_speeds.get(matched_id, 0)
                            smooth_speed = int(alpha * prev_speed + (1 - alpha) * raw_speed)

                            vehicle_speeds[matched_id] = smooth_speed

                        vehicle_tracks[matched_id] = current_center
                        speed = vehicle_speeds.get(matched_id, 0)

                        count += 1

                        # -------- EMERGENCY --------
                        label = names[int(cls)].lower()
                        if "ambulance" in label or "fire" in label or "police" in label:
                            emergency_lane = cam_id

                        # ✅ DRAW PERFECT BOX
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(im0, f"{names[int(cls)]} {conf:.2f}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 255), 2)

            lane_counts.append(count)
            frames.append(im0)

        if all_finished:
            print("🎉 All videos completed")
            break

        # -------- DYNAMIC TIMING --------
        max_vehicle_count = max(lane_counts) if max(lane_counts) > 0 else 1

        dynamic_green_times = []
        for count in lane_counts:
            green_time = MIN_GREEN_TIME + (MAX_GREEN_TIME - MIN_GREEN_TIME) * (count / max_vehicle_count)
            dynamic_green_times.append(int(green_time * FPS))

        # -------- PRIORITY --------
        priority_scores = [(lane_counts[i] * 2 + lane_wait_time[i]) for i in range(len(lane_counts))]

        # -------- SIGNAL LOGIC --------
        if emergency_lane != -1:
            current_green = emergency_lane
            green_timer = 0
        else:
            if current_green == -1:
                current_green = int(np.argmax(priority_scores))
                green_timer = 0
            else:
                green_timer += 1

                if green_timer > dynamic_green_times[current_green]:
                    current_green = int(np.argmax(priority_scores))
                    green_timer = 0

        green_lane = current_green

        # -------- WAIT TIME --------
        for i in range(len(lane_wait_time)):
            lane_wait_time[i] = 0 if i == green_lane else lane_wait_time[i] + 1

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

            cv2.putText(frame, f"Green Time: {dynamic_green_times[i]//FPS}s", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--view-img", action="store_true")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    print_args(vars(opt))
    return opt


def main(opt):
    run(opt.weights, opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        view_img=opt.view_img)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)