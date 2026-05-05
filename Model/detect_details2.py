# import argparse
# import os
# import sys
# import json
# from pathlib import Path
# import numpy as np
# import torch
# import cv2
# import threading
# from flask import Flask, Response

# from utils.augmentations import letterbox

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]

# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from models.common import DetectMultiBackend
# from utils.general import check_img_size, non_max_suppression
# from utils.torch_utils import select_device, smart_inference_mode


# # ===============================
# # SETTINGS
# # ===============================
# NUM_LANES = 4
# INITIAL_SCAN_TIME = 1
# MIN_GREEN_TIME = 15
# MAX_WAIT_TIME = 60
# FPS = 30
# JSON_INTERVAL = 10

# current_green = 0
# mode = "INITIAL_SCAN"
# mode_timer = 0

# lane_wait_time = [0] * NUM_LANES
# last_lane_counts = [0] * NUM_LANES

# # 🔥 LIVE VIDEO STORAGE
# latest_frames = [None] * NUM_LANES


# # ===============================
# # FLASK VIDEO SERVER
# # ===============================
# app = Flask(__name__)

# def generate_stream(lane_id):
#     global latest_frames

#     while True:
#         frame = latest_frames[lane_id]

#         if frame is None:
#             continue

#         # 🔥 resize for speed
#         frame = cv2.resize(frame, (640, 360))

#         # 🔥 compress
#         _, buffer = cv2.imencode(
#             '.jpg', frame,
#             [int(cv2.IMWRITE_JPEG_QUALITY), 60]
#         )

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' +
#                buffer.tobytes() + b'\r\n')


# @app.route('/video/<int:lane_id>')
# def video_feed(lane_id):
#     return Response(generate_stream(lane_id),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# def start_flask():
#     app.run(host='0.0.0.0', port=8000, threaded=True)


# # ===============================
# # DETECTION
# # ===============================
# def detect_vehicles(frame, model, device, imgsz, stride, names,
#                     conf_thres, iou_thres, half):

#     img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)

#     img = torch.from_numpy(img).to(device)
#     img = img.half() if half else img.float()
#     img /= 255.0
#     img = img.unsqueeze(0)

#     pred = model(img)
#     pred = non_max_suppression(pred, conf_thres, iou_thres)

#     count = 0
#     emergency = False

#     for det in pred:
#         if len(det):
#             for *xyxy, conf, cls in det:
#                 label = names[int(cls)].lower()

#                 if label == "ambulance":
#                     emergency = True

#                 count += 1

#     return count, emergency


# # ===============================
# # PRIORITY LOGIC
# # ===============================
# def get_best_lane(finished, emergency_lane):

#     if emergency_lane != -1 and not finished[emergency_lane]:
#         return emergency_lane

#     for i in range(NUM_LANES):
#         if lane_wait_time[i] >= MAX_WAIT_TIME * FPS:
#             return i

#     return int(np.argmax(last_lane_counts))


# # ===============================
# # MAIN
# # ===============================
# @smart_inference_mode()
# def run(weights, sources, imgsz=(640, 640),
#         conf_thres=0.35, iou_thres=0.45, device=""):

#     global current_green, mode, mode_timer
#     global lane_wait_time, last_lane_counts, latest_frames

#     device = select_device(device)

#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names

#     imgsz = check_img_size(imgsz, s=stride)

#     half = device.type != 'cpu'
#     if half:
#         model.model.half()

#     # 🔥 warmup
#     model(torch.zeros(1, 3, *imgsz).to(device))

#     caps = [cv2.VideoCapture(src) for src in sources]
#     finished = [False] * NUM_LANES

#     frame_count = 0

#     while True:

#         emergency_lane = -1

#         # ===============================
#         # READ FRAMES
#         # ===============================
#         frames = []
#         for i, cap in enumerate(caps):

#             ret, frame = cap.read()

#             if not ret:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 ret, frame = cap.read()

#             if ret:
#                 latest_frames[i] = frame.copy()  # 🔥 store for stream
#                 frames.append(frame)
#             else:
#                 frames.append(None)

#         # ===============================
#         # INITIAL SCAN
#         # ===============================
#         if mode == "INITIAL_SCAN":

#             for i in range(NUM_LANES):
#                 if frames[i] is None:
#                     continue

#                 count, emergency = detect_vehicles(
#                     frames[i], model, device, imgsz,
#                     stride, names, conf_thres, iou_thres, half
#                 )

#                 last_lane_counts[i] = count

#                 if emergency:
#                     emergency_lane = i

#             mode_timer += 1

#             if mode_timer >= INITIAL_SCAN_TIME * FPS:
#                 current_green = get_best_lane(finished, emergency_lane)
#                 mode = "GREEN"
#                 mode_timer = 0

#         # ===============================
#         # GREEN MODE
#         # ===============================
#         else:

#             if frames[current_green] is not None:
#                 count, emergency = detect_vehicles(
#                     frames[current_green], model, device, imgsz,
#                     stride, names, conf_thres, iou_thres, half
#                 )

#                 last_lane_counts[current_green] = count

#                 if emergency:
#                     emergency_lane = current_green

#             mode_timer += 1

#             if mode_timer >= MIN_GREEN_TIME * FPS:

#                 for i in range(NUM_LANES):
#                     if frames[i] is None:
#                         continue

#                     count, emergency = detect_vehicles(
#                         frames[i], model, device, imgsz,
#                         stride, names, conf_thres, iou_thres, half
#                     )

#                     last_lane_counts[i] = count

#                     if emergency:
#                         emergency_lane = i

#                 current_green = get_best_lane(finished, emergency_lane)
#                 mode_timer = 0

#         # ===============================
#         # WAIT TIME
#         # ===============================
#         for i in range(NUM_LANES):
#             if i == current_green:
#                 lane_wait_time[i] = 0
#             else:
#                 lane_wait_time[i] += 1
#                 lane_wait_time[i] = min(lane_wait_time[i], MAX_WAIT_TIME * FPS)

#         # ===============================
#         # JSON OUTPUT
#         # ===============================
#         if frame_count % JSON_INTERVAL == 0:
#             data = {
#                 "lane1": int(last_lane_counts[0]),
#                 "lane2": int(last_lane_counts[1]),
#                 "lane3": int(last_lane_counts[2]),
#                 "lane4": int(last_lane_counts[3]),
#                 "green_lane": int(current_green + 1),
#                 "wait": [int(x / FPS) for x in lane_wait_time],
#                 "emergency": emergency_lane != -1,
#                 "mode": mode
#             }

#             print(json.dumps(data), flush=True)

#         frame_count += 1


# # ===============================
# # ARGS
# # ===============================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--source", nargs='+', type=str, required=True)
#     parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
#     parser.add_argument("--conf-thres", type=float, default=0.35)
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
#     return opt


# def main(opt):
#     # 🔥 start video server in background
#     threading.Thread(target=start_flask, daemon=True).start()

#     run(
#         opt.weights,
#         opt.source,
#         imgsz=opt.imgsz,
#         conf_thres=opt.conf_thres
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
import threading
import time
from flask import Flask, Response

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes   # ✅ FIXED IMPORT
from utils.torch_utils import select_device, smart_inference_mode


# ===============================
# SETTINGS
# ===============================
NUM_LANES = 4
INITIAL_SCAN_TIME = 1
MIN_GREEN_TIME = 15
MAX_WAIT_TIME = 60
FPS = 30
JSON_INTERVAL = 10

current_green = 0
mode = "INITIAL_SCAN"
mode_timer = 0

lane_wait_time = [0] * NUM_LANES
last_lane_counts = [0] * NUM_LANES

latest_frames = [None] * NUM_LANES


# ===============================
# FLASK VIDEO SERVER
# ===============================
app = Flask(__name__)

def generate_stream(lane_id):
    global latest_frames

    while True:
        frame = latest_frames[lane_id]

        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (640, 360))

        _, buffer = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        )

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.route('/video/<int:lane_id>')
def video_feed(lane_id):
    return Response(generate_stream(lane_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_flask():
    app.run(host='0.0.0.0', port=8000, threaded=True)


# ===============================
# DETECTION (WITH BOXES)
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

            det[:, :4] = scale_boxes(
                img.shape[2:], det[:, :4], frame.shape
            ).round()

            for *xyxy, conf, cls in det:

                x1, y1, x2, y2 = map(int, xyxy)
                label = names[int(cls)].lower()

                if label == "ambulance":
                    emergency = True

                count += 1

                # 🔥 DRAW BOX
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                cv2.putText(frame, label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 2)

    return frame, count, emergency


# ===============================
# PRIORITY LOGIC
# ===============================
def get_best_lane(finished, emergency_lane):

    if emergency_lane != -1 and not finished[emergency_lane]:
        return emergency_lane

    for i in range(NUM_LANES):
        if lane_wait_time[i] >= MAX_WAIT_TIME * FPS:
            return i

    return int(np.argmax(last_lane_counts))


# ===============================
# MAIN
# ===============================
@smart_inference_mode()
def run(weights, sources, imgsz=(640, 640),
        conf_thres=0.35, iou_thres=0.45, device=""):

    global current_green, mode, mode_timer
    global lane_wait_time, last_lane_counts, latest_frames

    torch.backends.cudnn.benchmark = True

    device = select_device(device)

    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    imgsz = check_img_size(imgsz, s=stride)

    half = device.type != 'cpu'
    if half:
        model.model.half()

    # ✅ FIXED WARMUP (NO ERROR NOW)
    dummy = torch.zeros(1, 3, *imgsz).to(device)
    dummy = dummy.half() if half else dummy.float()
    model(dummy)

    caps = [cv2.VideoCapture(src) for src in sources]
    finished = [False] * NUM_LANES

    frame_count = 0

    while True:

        emergency_lane = -1
        frames = []

        # ===============================
        # READ FRAMES
        # ===============================
        for i, cap in enumerate(caps):

            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if ret:
                frames.append(frame)
            else:
                frames.append(None)

        # ===============================
        # INITIAL SCAN
        # ===============================
        if mode == "INITIAL_SCAN":

            for i in range(NUM_LANES):
                if frames[i] is None:
                    continue

                processed, count, emergency = detect_vehicles(
                    frames[i], model, device, imgsz,
                    stride, names, conf_thres, iou_thres, half
                )

                latest_frames[i] = processed   # ✅ store processed frame
                last_lane_counts[i] = count

                if emergency:
                    emergency_lane = i

            mode_timer += 1

            if mode_timer >= INITIAL_SCAN_TIME * FPS:
                current_green = get_best_lane(finished, emergency_lane)
                mode = "GREEN"
                mode_timer = 0

        # ===============================
        # GREEN MODE
        # ===============================
        else:

            for i in range(NUM_LANES):

                if frames[i] is None:
                    continue

                # only detect green lane
                if i == current_green:

                    processed, count, emergency = detect_vehicles(
                        frames[i], model, device, imgsz,
                        stride, names, conf_thres, iou_thres, half
                    )

                    last_lane_counts[i] = count

                    if emergency:
                        emergency_lane = i

                else:
                    processed = frames[i]

                latest_frames[i] = processed   # ✅ store with boxes

            mode_timer += 1

            if mode_timer >= MIN_GREEN_TIME * FPS:

                for i in range(NUM_LANES):
                    if frames[i] is None:
                        continue

                    processed, count, emergency = detect_vehicles(
                        frames[i], model, device, imgsz,
                        stride, names, conf_thres, iou_thres, half
                    )

                    latest_frames[i] = processed
                    last_lane_counts[i] = count

                    if emergency:
                        emergency_lane = i

                current_green = get_best_lane(finished, emergency_lane)
                mode_timer = 0

        # ===============================
        # WAIT TIME
        # ===============================
        for i in range(NUM_LANES):
            if i == current_green:
                lane_wait_time[i] = 0
            else:
                lane_wait_time[i] += 1
                lane_wait_time[i] = min(lane_wait_time[i], MAX_WAIT_TIME * FPS)

        # ===============================
        # JSON OUTPUT
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

    threading.Thread(target=start_flask, daemon=True).start()

    run(
        opt.weights,
        opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)