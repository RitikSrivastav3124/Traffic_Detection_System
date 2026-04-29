# import argparse
# import os
# import sys
# from pathlib import Path
# import numpy as np
# import torch
# import cv2

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]

# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from models.common import DetectMultiBackend
# from utils.general import check_img_size, non_max_suppression, print_args, scale_boxes
# from utils.torch_utils import select_device, smart_inference_mode

# # -------- TRACKING --------
# vehicle_tracks = {}
# vehicle_speeds = {}
# next_vehicle_id = 0

# pixels_per_meter = 155
# max_tracking_distance = 50

# # -------- SIGNAL SYSTEM --------
# lane_wait_time = [0, 0, 0, 0]
# green_timer = 0
# current_green = -1

# MIN_GREEN_TIME = 2
# MAX_GREEN_TIME = 3
# FPS = 30


# @smart_inference_mode()
# def run(weights, sources, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, device="", view_img=True):

#     global next_vehicle_id, lane_wait_time, green_timer, current_green

#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names

#     imgsz = check_img_size(imgsz, s=stride)

#     caps = []
#     finished = [False] * len(sources)

#     for src in sources:
#         cap = cv2.VideoCapture(src)
#         if not cap.isOpened():
#             print(f"❌ Cannot open {src}")
#         else:
#             print(f"✅ Loaded {src}")
#         caps.append(cap)

#     # -------- REAL FPS --------
#     REAL_FPS = caps[0].get(cv2.CAP_PROP_FPS)
#     if REAL_FPS == 0 or REAL_FPS is None:
#         REAL_FPS = 30

#     print(f"🎯 Using FPS: {REAL_FPS}")

#     # -------- OUTPUT --------
#     save_path = "output"
#     os.makedirs(save_path, exist_ok=True)

#     writers = []
#     for i, cap in enumerate(caps):
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
#         fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

#         out = cv2.VideoWriter(
#             f"{save_path}/lane{i+1}.mp4",
#             cv2.VideoWriter_fourcc(*'mp4v'),
#             fps,
#             (width, height)
#         )
#         writers.append(out)

#     while True:

#         frames = []
#         lane_counts = []
#         all_finished = True
#         emergency_lane = -1

#         for cam_id, cap in enumerate(caps):

#             if finished[cam_id]:
#                 frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
#                 lane_counts.append(0)
#                 continue

#             ret, frame = cap.read()

#             if not ret:
#                 print(f"✅ Video finished: {sources[cam_id]}")
#                 finished[cam_id] = True
#                 frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
#                 lane_counts.append(0)
#                 continue
#             else:
#                 all_finished = False

#             im0 = frame.copy()

#             # -------- PREPROCESS --------
#             img = cv2.resize(im0, (imgsz[0], imgsz[1]))
#             img = img[:, :, ::-1].transpose(2, 0, 1)
#             img = np.ascontiguousarray(img)

#             img = torch.from_numpy(img).to(device).float() / 255.0
#             img = img.unsqueeze(0)

#             # -------- INFERENCE --------
#             pred = model(img)
#             pred = non_max_suppression(pred, conf_thres, iou_thres)

#             count = 0

#             for det in pred:
#                 if len(det):

#                     det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape)

#                     for *xyxy, conf, cls in det:

#                         x1, y1, x2, y2 = map(int, xyxy)

#                         cx = int((x1 + x2) / 2)
#                         cy = int((y1 + y2) / 2)

#                         current_center = np.array([cx, cy])
#                         matched_id = None

#                         # -------- TRACKING --------
#                         for vid, prev_center in vehicle_tracks.items():
#                             distance = np.linalg.norm(current_center - prev_center)
#                             if distance < max_tracking_distance:
#                                 matched_id = vid
#                                 break

#                         if matched_id is None:
#                             matched_id = next_vehicle_id
#                             next_vehicle_id += 1
#                             vehicle_speeds[matched_id] = 0

#                         if matched_id in vehicle_tracks:
#                             prev_center = vehicle_tracks[matched_id]

#                             pixel_distance = np.linalg.norm(current_center - prev_center)
#                             distance_m = pixel_distance / pixels_per_meter

#                             # -------- FIXED SPEED --------
#                             raw_speed = distance_m * REAL_FPS * 3.6

#                             alpha = 0.8
#                             prev_speed = vehicle_speeds.get(matched_id, 0)
#                             smooth_speed = int(alpha * prev_speed + (1 - alpha) * raw_speed)

#                             vehicle_speeds[matched_id] = smooth_speed

#                         vehicle_tracks[matched_id] = current_center
#                         speed = vehicle_speeds.get(matched_id, 0)

#                         count += 1

#                         # -------- EMERGENCY --------
#                         label = names[int(cls)].lower()
#                         if "ambulance" in label:
#                             emergency_lane = cam_id

#                         # -------- DRAW --------
#                         cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.putText(im0, f"ID:{matched_id} {speed}km/h",
#                                     (x1, y1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX,
#                                     0.6, (0, 255, 255), 2)

#             lane_counts.append(count)
#             frames.append(im0)

#         if all_finished:
#             print("🎉 All videos completed")
#             break

#         # -------- SIGNAL LOGIC --------
#         priority_scores = []
#         for i in range(len(lane_counts)):
#             priority = (lane_counts[i] * 2) + (lane_wait_time[i] * 1)
#             priority_scores.append(priority)

#         if emergency_lane != -1:
#             current_green = emergency_lane
#             green_timer = 0
#         else:
#             if current_green == -1:
#                 current_green = int(np.argmax(priority_scores))
#                 green_timer = 0
#             else:
#                 green_timer += 1

#                 if green_timer > MIN_GREEN_TIME * FPS:
#                     best_lane = int(np.argmax(priority_scores))

#                     if best_lane != current_green or green_timer > MAX_GREEN_TIME * FPS:
#                         current_green = best_lane
#                         green_timer = 0

#         green_lane = current_green

#         # -------- WAIT UPDATE --------
#         for i in range(len(lane_wait_time)):
#             if i == green_lane:
#                 lane_wait_time[i] = 0
#             else:
#                 lane_wait_time[i] += 1

#         signals = ["RED"] * len(frames)
#         signals[green_lane] = "GREEN"

#         # -------- DISPLAY --------
#         for i, frame in enumerate(frames):

#             color = (0, 255, 0) if signals[i] == "GREEN" else (0, 0, 255)

#             cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#             cv2.putText(frame, f"Signal: {signals[i]}", (20, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

#             cv2.putText(frame, f"Wait: {lane_wait_time[i]//FPS}s", (20, 120),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#             if view_img:
#                 cv2.imshow(f"Camera {i+1}", frame)

#             writers[i].write(frame)

#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     for cap in caps:
#         cap.release()

#     for w in writers:
#         w.release()

#     cv2.destroyAllWindows()


# def parse_opt():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", nargs='+', type=str, required=True)
#     parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--view-img", action="store_true")

#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

#     print_args(vars(opt))
#     return opt


# def main(opt):
#     run(opt.weights, opt.source,
#         imgsz=opt.imgsz,
#         conf_thres=opt.conf_thres,
#         view_img=opt.view_img)


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, print_args, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

# -------- TRACKING (PER LANE) --------
NUM_LANES = 4

vehicle_tracks = [{} for _ in range(NUM_LANES)]
vehicle_speeds = [{} for _ in range(NUM_LANES)]
vehicle_last_seen = [{} for _ in range(NUM_LANES)]
next_vehicle_id = [0] * NUM_LANES

max_tracking_distance = 80
max_missing_frames = 15

# -------- SIGNAL SYSTEM --------
lane_wait_time = [0] * NUM_LANES
green_timer = 0
current_green = -1

MIN_GREEN_TIME = 2
MAX_GREEN_TIME = 5
FPS = 30


@smart_inference_mode()
def run(weights, sources, imgsz=(640, 640), conf_thres=0.35, iou_thres=0.45, device="", view_img=True):

    global next_vehicle_id, lane_wait_time, green_timer, current_green

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    imgsz = check_img_size(imgsz, s=stride)

    caps = [cv2.VideoCapture(src) for src in sources]

    REAL_FPS = caps[0].get(cv2.CAP_PROP_FPS) or 30

    os.makedirs("output", exist_ok=True)

    writers = []
    for i, cap in enumerate(caps):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        writers.append(cv2.VideoWriter(
            f"output/lane{i+1}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        ))

    frame_count = 0

    while True:
        frames = []
        lane_counts = []
        all_finished = True
        emergency_lane = -1

        for cam_id, cap in enumerate(caps):

            ret, frame = cap.read()

            if not ret:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                lane_counts.append(0)
                continue

            all_finished = False
            im0 = frame.copy()

            # -------- PREPROCESS --------
            img = letterbox(im0, imgsz, stride=stride, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float() / 255.0
            img = img.unsqueeze(0)

            # -------- INFERENCE --------
            pred = model(img)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            count = 0
            current_ids = set()

            for det in pred:
                if len(det):

                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det:

                        x1, y1, x2, y2 = map(int, xyxy)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        center = np.array([cx, cy])

                        matched_id = None
                        min_dist = 9999

                        # ✅ FIX: PER LANE TRACKING
                        for vid, prev_center in vehicle_tracks[cam_id].items():
                            dist = np.linalg.norm(center - prev_center)
                            if dist < max_tracking_distance and dist < min_dist:
                                matched_id = vid
                                min_dist = dist

                        if matched_id is None:
                            matched_id = next_vehicle_id[cam_id]
                            next_vehicle_id[cam_id] += 1
                            vehicle_speeds[cam_id][matched_id] = 0

                        # -------- SPEED --------
                        if matched_id in vehicle_tracks[cam_id]:
                            prev = vehicle_tracks[cam_id][matched_id]
                            pixel_dist = np.linalg.norm(center - prev)
                            speed = int((pixel_dist / 155) * REAL_FPS * 3.6)

                            vehicle_speeds[cam_id][matched_id] = int(
                                0.7 * vehicle_speeds[cam_id].get(matched_id, 0) + 0.3 * speed
                            )

                        vehicle_tracks[cam_id][matched_id] = center
                        vehicle_last_seen[cam_id][matched_id] = frame_count
                        current_ids.add(matched_id)

                        speed = vehicle_speeds[cam_id].get(matched_id, 0)
                        label = names[int(cls)].lower()

                        if label == "ambulance":
                            emergency_lane = cam_id

                        count += 1

                        # -------- DRAW --------
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(im0,
                                    f"{label} ID:{matched_id} {speed}km/h",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2)

            # -------- REMOVE LOST TRACKS --------
            to_delete = []
            for vid in vehicle_last_seen[cam_id]:
                if frame_count - vehicle_last_seen[cam_id][vid] > max_missing_frames:
                    to_delete.append(vid)

            for vid in to_delete:
                vehicle_tracks[cam_id].pop(vid, None)
                vehicle_speeds[cam_id].pop(vid, None)
                vehicle_last_seen[cam_id].pop(vid, None)

            lane_counts.append(count)
            frames.append(im0)

        if all_finished:
            break

        frame_count += 1

        # -------- SIGNAL --------
        priority = [(lane_counts[i]*2 + lane_wait_time[i]) for i in range(NUM_LANES)]

        if emergency_lane != -1:
            current_green = emergency_lane
            green_timer = 0
        else:
            if current_green == -1:
                current_green = int(np.argmax(priority))
            else:
                green_timer += 1
                if green_timer > MIN_GREEN_TIME * FPS:
                    best = int(np.argmax(priority))
                    if best != current_green or green_timer > MAX_GREEN_TIME * FPS:
                        current_green = best
                        green_timer = 0

        for i in range(NUM_LANES):
            lane_wait_time[i] = 0 if i == current_green else lane_wait_time[i] + 1

        signals = ["RED"] * NUM_LANES
        signals[current_green] = "GREEN"

        # -------- DISPLAY --------
        for i, frame in enumerate(frames):

            color = (0,255,0) if signals[i]=="GREEN" else (0,0,255)

            cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.putText(frame, f"Signal: {signals[i]}", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,1,color,3)

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
    run(opt.weights, opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        view_img=opt.view_img)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)