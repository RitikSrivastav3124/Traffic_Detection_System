
# import argparse
# import os
# import sys
# import json
# from pathlib import Path
# import numpy as np
# import torch
# import cv2
# from utils.augmentations import letterbox

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]

# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from models.common import DetectMultiBackend
# from utils.general import check_img_size, non_max_suppression, print_args, scale_boxes
# from utils.torch_utils import select_device, smart_inference_mode


# # =====================================
# # SETTINGS
# # =====================================
# NUM_LANES = 4

# INITIAL_SCAN_TIME = 1
# MIN_GREEN_TIME = 15
# MAX_WAIT_TIME = 60
# FPS = 30

# current_green = 0
# mode = "INITIAL_SCAN"
# mode_timer = 0

# lane_wait_time = [0] * NUM_LANES
# last_lane_counts = [0] * NUM_LANES


# # =====================================
# # DETECTION
# # =====================================
# def detect_vehicles(frame, model, device, imgsz, stride, names,
#                     conf_thres, iou_thres):

#     im0 = frame.copy()

#     img = letterbox(im0, imgsz, stride=stride, auto=True)[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)

#     img = torch.from_numpy(img).to(device).float() / 255.0
#     img = img.unsqueeze(0)

#     pred = model(img)
#     pred = non_max_suppression(pred, conf_thres, iou_thres)

#     count = 0
#     emergency = False

#     for det in pred:

#         if len(det):

#             det[:, :4] = scale_boxes(
#                 img.shape[2:],
#                 det[:, :4],
#                 im0.shape
#             ).round()

#             for *xyxy, conf, cls in det:

#                 x1, y1, x2, y2 = map(int, xyxy)

#                 label = names[int(cls)].lower()

#                 if label == "ambulance":
#                     emergency = True

#                 count += 1

#                 cv2.rectangle(im0, (x1, y1), (x2, y2),
#                               (0, 255, 0), 2)

#                 cv2.putText(im0, label,
#                             (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.6, (0, 255, 255), 2)

#     return im0, count, emergency


# # =====================================
# # PRIORITY LOGIC
# # =====================================
# def get_best_lane(finished, emergency_lane):

#     global lane_wait_time, last_lane_counts

#     if emergency_lane != -1 and not finished[emergency_lane]:
#         return emergency_lane

#     max_wait_frames = MAX_WAIT_TIME * FPS

#     for i in range(NUM_LANES):
#         if not finished[i] and lane_wait_time[i] >= max_wait_frames:
#             return i

#     best_lane = -1
#     best_count = -1

#     for i in range(NUM_LANES):
#         if finished[i]:
#             continue

#         if last_lane_counts[i] > best_count:
#             best_count = last_lane_counts[i]
#             best_lane = i

#     return best_lane


# # =====================================
# # MAIN
# # =====================================
# @smart_inference_mode()
# def run(weights,
#         sources,
#         imgsz=(640, 640),
#         conf_thres=0.35,
#         iou_thres=0.45,
#         device="",
#         view_img=True):

#     global current_green, mode, mode_timer
#     global lane_wait_time, last_lane_counts

#     device = select_device(device)

#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names

#     imgsz = check_img_size(imgsz, s=stride)

#     caps = [cv2.VideoCapture(src) for src in sources]

#     os.makedirs("output", exist_ok=True)

#     writers = []
#     paused_frames = []
#     finished = [False] * NUM_LANES

#     for i, cap in enumerate(caps):

#         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
#         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
#         fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

#         writers.append(
#             cv2.VideoWriter(
#                 f"output/lane{i+1}.mp4",
#                 cv2.VideoWriter_fourcc(*'mp4v'),
#                 fps,
#                 (w, h)
#             )
#         )

#         paused_frames.append(np.zeros((h, w, 3), dtype=np.uint8))

#     while True:

#         frames = []
#         emergency_lane = -1

#         # =====================================
#         # INITIAL SCAN
#         # =====================================
#         if mode == "INITIAL_SCAN":

#             for cam_id, cap in enumerate(caps):

#                 if finished[cam_id]:
#                     frames.append(paused_frames[cam_id].copy())
#                     continue

#                 ret, frame = cap.read()

#                 if not ret:
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                     ret, frame = cap.read()

#                 if not ret:
#                     frames.append(paused_frames[cam_id].copy())
#                     continue

#                 processed, count, emergency = detect_vehicles(
#                     frame, model, device, imgsz,
#                     stride, names,
#                     conf_thres, iou_thres
#                 )

#                 if emergency:
#                     emergency_lane = cam_id

#                 last_lane_counts[cam_id] = count
#                 paused_frames[cam_id] = processed.copy()
#                 frames.append(processed)

#             mode_timer += 1

#             if mode_timer >= INITIAL_SCAN_TIME * FPS:
#                 current_green = get_best_lane(finished, emergency_lane)
#                 mode = "GREEN"
#                 mode_timer = 0

#         # =====================================
#         # GREEN MODE
#         # =====================================
#         else:

#             for cam_id, cap in enumerate(caps):

#                 if finished[cam_id]:
#                     frames.append(paused_frames[cam_id].copy())
#                     continue

#                 if cam_id != current_green:
#                     frames.append(paused_frames[cam_id].copy())
#                     continue

#                 ret, frame = cap.read()

#                 if not ret:
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                     ret, frame = cap.read()

#                 if not ret:
#                     frames.append(paused_frames[cam_id].copy())
#                     continue

#                 processed, count, emergency = detect_vehicles(
#                     frame, model, device, imgsz,
#                     stride, names,
#                     conf_thres, iou_thres
#                 )

#                 if emergency:
#                     emergency_lane = cam_id

#                 last_lane_counts[cam_id] = count
#                 paused_frames[cam_id] = processed.copy()
#                 frames.append(processed)

#             mode_timer += 1

#             if mode_timer >= MIN_GREEN_TIME * FPS:

#                 for cam_id, cap in enumerate(caps):

#                     if finished[cam_id]:
#                         continue

#                     ret, frame = cap.read()

#                     if not ret:
#                         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                         ret, frame = cap.read()

#                     if not ret:
#                         continue

#                     processed, count, emergency = detect_vehicles(
#                         frame, model, device, imgsz,
#                         stride, names,
#                         conf_thres, iou_thres
#                     )

#                     paused_frames[cam_id] = processed.copy()
#                     last_lane_counts[cam_id] = count

#                     if emergency:
#                         emergency_lane = cam_id

#                 current_green = get_best_lane(finished, emergency_lane)
#                 mode_timer = 0

#         # =====================================
#         # WAITING TIME
#         # =====================================
#         for i in range(NUM_LANES):

#             if finished[i]:
#                 continue

#             if i == current_green:
#                 lane_wait_time[i] = 0
#             else:
#                 lane_wait_time[i] += 1

#                 if lane_wait_time[i] > MAX_WAIT_TIME * FPS:
#                     lane_wait_time[i] = MAX_WAIT_TIME * FPS

#         # =====================================
#         # EXIT
#         # =====================================
#         if all(finished):
#             break

#         # =====================================
#         # SIGNALS
#         # =====================================
#         signals = ["RED"] * NUM_LANES
#         signals[current_green] = "GREEN"

#         # =====================================
#         # JSON OUTPUT FOR NODE BACKEND
#         # =====================================
#         if mode_timer % FPS == 0:
#             data = {
#                 "lane1": int(last_lane_counts[0]),
#                 "lane2": int(last_lane_counts[1]),
#                 "lane3": int(last_lane_counts[2]),
#                 "lane4": int(last_lane_counts[3]),
#                 "green_lane": int(current_green + 1),
#                 "wait": [
#                     int(lane_wait_time[0] / FPS),
#                     int(lane_wait_time[1] / FPS),
#                     int(lane_wait_time[2] / FPS),
#                     int(lane_wait_time[3] / FPS)
#                 ],
#                 "emergency": emergency_lane != -1,
#                 "mode": mode
#             }

#             print(json.dumps(data), flush=True)

#         # =====================================
#         # DISPLAY
#         # =====================================
#         for i, frame in enumerate(frames):

#             color = (0, 255, 0) if signals[i] == "GREEN" else (0, 0, 255)

#             cv2.putText(frame, f"Lane {i+1}",
#                         (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (255, 255, 255), 2)

#             cv2.putText(frame,
#                         f"Vehicles: {last_lane_counts[i]}",
#                         (20, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8, (255, 255, 255), 2)

#             cv2.putText(frame,
#                         f"Wait: {lane_wait_time[i]//FPS}s",
#                         (20, 120),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8, (255, 255, 0), 2)

#             cv2.putText(frame,
#                         f"Signal: {signals[i]}",
#                         (20, 160),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1, color, 3)

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


# # =====================================
# # ARGUMENTS
# # =====================================
# def parse_opt():

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", nargs='+', type=str, required=True)
#     parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
#     parser.add_argument("--conf-thres", type=float, default=0.35)
#     parser.add_argument("--view-img", action="store_true")

#     opt = parser.parse_args()

#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

#     print_args(vars(opt))
#     return opt


# def main(opt):

#     run(
#         opt.weights,
#         opt.source,
#         imgsz=opt.imgsz,
#         conf_thres=opt.conf_thres,
#         view_img=opt.view_img
#     )


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)


import argparse
import os
import sys
import json
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


# ===============================
# SETTINGS
# ===============================
NUM_LANES = 4
INITIAL_SCAN_TIME = 1
MIN_GREEN_TIME = 15
MAX_WAIT_TIME = 60
FPS = 30

# 🔥 JSON output fast kar diya
JSON_INTERVAL = 10   # frames (~0.3 sec)

current_green = 0
mode = "INITIAL_SCAN"
mode_timer = 0

lane_wait_time = [0] * NUM_LANES
last_lane_counts = [0] * NUM_LANES


# ===============================
# DETECTION (OPTIMIZED)
# ===============================
def detect_vehicles(frame, model, device, imgsz, stride, names,
                    conf_thres, iou_thres, half):

    img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    img = img.unsqueeze(0)

    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    count = 0
    emergency = False

    for det in pred:
        if len(det):
            for *xyxy, conf, cls in det:
                label = names[int(cls)].lower()

                if label == "ambulance":
                    emergency = True

                count += 1

    return count, emergency


# ===============================
# PRIORITY LOGIC
# ===============================
def get_best_lane(finished, emergency_lane):
    global lane_wait_time, last_lane_counts

    if emergency_lane != -1 and not finished[emergency_lane]:
        return emergency_lane

    max_wait_frames = MAX_WAIT_TIME * FPS

    for i in range(NUM_LANES):
        if not finished[i] and lane_wait_time[i] >= max_wait_frames:
            return i

    return int(np.argmax(last_lane_counts))


# ===============================
# MAIN
# ===============================
@smart_inference_mode()
def run(weights,
        sources,
        imgsz=(640, 640),
        conf_thres=0.35,
        iou_thres=0.45,
        device="",
        view_img=False):

    global current_green, mode, mode_timer
    global lane_wait_time, last_lane_counts

    device = select_device(device)

    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    imgsz = check_img_size(imgsz, s=stride)

    # 🔥 GPU FAST MODE
    half = device.type != 'cpu'
    if half:
        model.model.half()

    # 🔥 WARMUP (important for speed)
    model(torch.zeros(1, 3, *imgsz).to(device))

    caps = [cv2.VideoCapture(src) for src in sources]
    finished = [False] * NUM_LANES

    frame_count = 0

    while True:

        emergency_lane = -1

        # ===============================
        # READ ALL FRAMES ONCE
        # ===============================
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frames.append(frame if ret else None)

        # ===============================
        # INITIAL SCAN
        # ===============================
        if mode == "INITIAL_SCAN":

            for i in range(NUM_LANES):
                if frames[i] is None:
                    continue

                count, emergency = detect_vehicles(
                    frames[i], model, device, imgsz,
                    stride, names, conf_thres, iou_thres, half
                )

                last_lane_counts[i] = count

                if emergency:
                    emergency_lane = i

            mode_timer += 1

            if mode_timer >= INITIAL_SCAN_TIME * FPS:
                current_green = get_best_lane(finished, emergency_lane)
                mode = "GREEN"
                mode_timer = 0

        # ===============================
        # GREEN MODE (ONLY 1 LANE DETECT)
        # ===============================
        else:

            if frames[current_green] is not None:
                count, emergency = detect_vehicles(
                    frames[current_green], model, device, imgsz,
                    stride, names, conf_thres, iou_thres, half
                )

                last_lane_counts[current_green] = count

                if emergency:
                    emergency_lane = current_green

            mode_timer += 1

            if mode_timer >= MIN_GREEN_TIME * FPS:

                for i in range(NUM_LANES):
                    if frames[i] is None:
                        continue

                    count, emergency = detect_vehicles(
                        frames[i], model, device, imgsz,
                        stride, names, conf_thres, iou_thres, half
                    )

                    last_lane_counts[i] = count

                    if emergency:
                        emergency_lane = i

                current_green = get_best_lane(finished, emergency_lane)
                mode_timer = 0

        # ===============================
        # WAIT TIME UPDATE
        # ===============================
        for i in range(NUM_LANES):
            if i == current_green:
                lane_wait_time[i] = 0
            else:
                lane_wait_time[i] += 1
                lane_wait_time[i] = min(lane_wait_time[i], MAX_WAIT_TIME * FPS)

        # ===============================
        # FAST JSON OUTPUT
        # ===============================
        if frame_count % JSON_INTERVAL == 0:
            data = {
                "lane1": int(last_lane_counts[0]),
                "lane2": int(last_lane_counts[1]),
                "lane3": int(last_lane_counts[2]),
                "lane4": int(last_lane_counts[3]),
                "green_lane": int(current_green + 1),
                "wait": [int(x / FPS) for x in lane_wait_time],
                "emergency": emergency_lane != -1,
                "mode": mode
            }

            print(json.dumps(data), flush=True)

        frame_count += 1


# ===============================
# ARGS
# ===============================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", nargs='+', type=str, required=True)
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.35)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    run(
        opt.weights,
        opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)