# import argparse
# import os
# import sys
# from pathlib import Path
# import numpy as np
# import torch
# import cv2
# import time

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]

# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages
# from utils.general import (
#     LOGGER,
#     check_img_size,
#     increment_path,
#     non_max_suppression,
#     print_args,
#     scale_boxes,
# )
# from utils.torch_utils import select_device, smart_inference_mode


# heatmap = None

# # -------- TRACKING --------
# vehicle_tracks = {}
# vehicle_speeds = {}
# next_vehicle_id = 0

# pixels_per_meter = 35
# max_tracking_distance = 50


# @smart_inference_mode()
# def run(
#         weights,
#         source,
#         imgsz=(640, 640),
#         conf_thres=0.25,
#         iou_thres=0.45,
#         device="",
#         view_img=False,
#         project=ROOT / "runs/detect",
#         name="exp",
#         exist_ok=False
# ):

#     global heatmap, next_vehicle_id

#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     device = select_device(device)

#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names

#     imgsz = check_img_size(imgsz, s=stride)

#     dataset = LoadImages(source, img_size=imgsz, stride=stride)

#     vid_path, vid_writer = None, None

#     model.warmup(imgsz=(1, 3, *imgsz))

#     for path, im, im0s, vid_cap, s in dataset:

#         start = time.time()

#         im = torch.from_numpy(im).to(model.device)
#         im = im.float() / 255.0

#         if len(im.shape) == 3:
#             im = im[None]

#         pred = model(im)
#         pred = non_max_suppression(pred, conf_thres, iou_thres)

#         for det in pred:

#             im0 = im0s.copy()
#             H, W, _ = im0.shape

#             if heatmap is None:
#                 heatmap = np.zeros((H, W), dtype=np.float32)

#             # ---------- LANES ----------
#             lane1 = np.array([[int(W*0.05), H],
#                               [int(W*0.30), H],
#                               [int(W*0.30), int(H*0.30)],
#                               [int(W*0.05), int(H*0.30)]], np.int32)

#             lane2 = np.array([[int(W*0.30), H],
#                               [int(W*0.55), H],
#                               [int(W*0.55), int(H*0.30)],
#                               [int(W*0.30), int(H*0.30)]], np.int32)

#             lane3 = np.array([[int(W*0.55), H],
#                               [int(W*0.80), H],
#                               [int(W*0.80), int(H*0.30)],
#                               [int(W*0.55), int(H*0.30)]], np.int32)

#             lane4 = np.array([[int(W*0.80), H],
#                               [int(W*0.98), H],
#                               [int(W*0.98), int(H*0.30)],
#                               [int(W*0.80), int(H*0.30)]], np.int32)

#             lanes = [lane1, lane2, lane3, lane4]
#             lane_colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]

#             for i,lane in enumerate(lanes):
#                 cv2.polylines(im0,[lane],True,lane_colors[i],3)

#             lane_counts = [0,0,0,0]
#             vehicle_total = 0

#             if len(det):

#                 det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape)

#                 for *xyxy, conf, cls in det:

#                     x1,y1,x2,y2 = map(int,xyxy)

#                     cx = int((x1+x2)/2)
#                     cy = int((y1+y2)/2)

#                     if cy >= H or cx >= W:
#                         continue

#                     heatmap[cy,cx] += 1
#                     vehicle_total += 1

#                     label = names[int(cls)]

#                     current_center = np.array([cx,cy])
#                     matched_id = None

#                     for vid,prev_center in vehicle_tracks.items():

#                         distance = np.linalg.norm(current_center-prev_center)

#                         if distance < max_tracking_distance:
#                             matched_id = vid
#                             break

#                     if matched_id is None:
#                         matched_id = next_vehicle_id
#                         next_vehicle_id += 1
#                         vehicle_speeds[matched_id] = 0

#                     if matched_id in vehicle_tracks:

#                         prev_center = vehicle_tracks[matched_id]

#                         pixel_distance = np.linalg.norm(current_center-prev_center)

#                         distance_m = pixel_distance / pixels_per_meter

#                         fps = max(1/(time.time()-start),1)

#                         time_sec = 1/fps

#                         speed = (distance_m/time_sec)*3.6

#                         vehicle_speeds[matched_id] = int(speed)

#                     vehicle_tracks[matched_id] = current_center

#                     speed = vehicle_speeds.get(matched_id,0)

#                     # ---------- DRAW ----------
#                     cv2.rectangle(im0,(x1,y1),(x2,y2),(0,255,0),2)

#                     cv2.putText(im0,
#                                 f"{label} ID:{matched_id} {speed} km/h",
#                                 (x1,y1-10),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.6,
#                                 (0,255,255),
#                                 2)

#                     cv2.circle(im0,(cx,cy),4,(0,255,255),-1)

#                     for idx,lane in enumerate(lanes):
#                         if cv2.pointPolygonTest(lane,(cx,cy),False)>=0:
#                             lane_counts[idx]+=1

#             fps=int(1/(time.time()-start))

#             cv2.putText(im0,f"FPS {fps}",(W-150,40),
#                         cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

#             for idx,count in enumerate(lane_counts):

#                 if count<3:
#                     density="LOW"
#                     color=(0,255,0)

#                 elif count<7:
#                     density="MEDIUM"
#                     color=(0,255,255)

#                 else:
#                     density="HIGH"
#                     color=(0,0,255)

#                 cv2.putText(im0,
#                             f"Lane {idx+1}: {count} {density}",
#                             (40,80+idx*40),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1,
#                             color,
#                             2)

#             cv2.rectangle(im0,(W-260,80),(W-20,150),(0,0,0),-1)

#             cv2.putText(im0,f"Vehicles: {vehicle_total}",
#                         (W-240,120),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.9,(0,255,0),2)

#             heatmap_norm=cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX)
#             heatmap_color=cv2.applyColorMap(
#                 heatmap_norm.astype(np.uint8),
#                 cv2.COLORMAP_JET
#             )

#             im0=cv2.addWeighted(im0,0.85,heatmap_color,0.15,0)

#             if view_img:
#                 cv2.imshow("Traffic Detection",im0)
#                 cv2.waitKey(1)

#             save_path=str(save_dir/Path(path).name)

#             if vid_path!=save_path:

#                 vid_path=save_path

#                 if isinstance(vid_writer,cv2.VideoWriter):
#                     vid_writer.release()

#                 if vid_cap:
#                     fps=vid_cap.get(cv2.CAP_PROP_FPS)
#                     w=int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                     h=int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 else:
#                     fps=30
#                     h,w=im0.shape[:2]

#                 save_path=str(Path(save_path).with_suffix(".mp4"))

#                 vid_writer=cv2.VideoWriter(
#                     save_path,
#                     cv2.VideoWriter_fourcc(*"mp4v"),
#                     fps,
#                     (w,h)
#                 )

#             vid_writer.write(im0)

#     LOGGER.info(f"Results saved to {save_dir}")


# def parse_opt():

#     parser=argparse.ArgumentParser()

#     parser.add_argument("--weights",type=str)
#     parser.add_argument("--source",type=str)
#     parser.add_argument("--imgsz",nargs="+",type=int,default=[640])
#     parser.add_argument("--conf-thres",type=float,default=0.25)
#     parser.add_argument("--view-img",action="store_true")

#     opt=parser.parse_args()

#     opt.imgsz*=2 if len(opt.imgsz)==1 else 1

#     print_args(vars(opt))

#     return opt


# def main(opt):
#     run(**vars(opt))


# if __name__=="__main__":

#     opt=parse_opt()
#     main(opt)


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

# pixels_per_meter = 35
# max_tracking_distance = 50


# @smart_inference_mode()
# def run(weights, sources, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, device="", view_img=True):

#     global next_vehicle_id

#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names

#     imgsz = check_img_size(imgsz, s=stride)

#     # -------- VIDEO LOAD --------
#     caps = []
#     for src in sources:
#         cap = cv2.VideoCapture(src)
#         if not cap.isOpened():
#             print(f"❌ Cannot open {src}")
#         else:
#             print(f"✅ Loaded {src}")
#         caps.append(cap)

#     while True:

#         frames = []
#         lane_counts = []

#         for cam_id, cap in enumerate(caps):

#             ret, frame = cap.read()
#             if not ret:
#                 print(f"⚠️ No frame from {sources[cam_id]}")
#                 continue

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

#                             speed = distance_m * 3.6 * 30
#                             vehicle_speeds[matched_id] = int(speed)

#                         vehicle_tracks[matched_id] = current_center
#                         speed = vehicle_speeds.get(matched_id, 0)

#                         count += 1

#                         # -------- DRAW --------
#                         cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.putText(im0, f"ID:{matched_id} {speed}km/h",
#                                     (x1, y1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX,
#                                     0.6, (0, 255, 255), 2)

#             lane_counts.append(count)
#             frames.append(im0)

#         if len(frames) == 0:
#             print("❌ No frames received from any video")
#             break

#         # -------- SIGNAL LOGIC --------
#         max_lane = np.argmax(lane_counts)
#         signals = ["RED"] * len(frames)
#         signals[max_lane] = "GREEN"

#         # -------- DISPLAY --------
#         for i, frame in enumerate(frames):

#             color = (0, 255, 0) if signals[i] == "GREEN" else (0, 0, 255)

#             cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}",
#                         (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         (255, 255, 255),
#                         2)

#             cv2.putText(frame, f"Signal: {signals[i]}",
#                         (20, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         color,
#                         3)

#             if view_img:
#                 cv2.imshow(f"Camera {i+1}", frame)

#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     for cap in caps:
#         cap.release()

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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, print_args, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

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
MAX_GREEN_TIME = 3
FPS = 30


@smart_inference_mode()
def run(weights, sources, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, device="", view_img=True):

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

    # -------- REAL FPS --------
    REAL_FPS = caps[0].get(cv2.CAP_PROP_FPS)
    if REAL_FPS == 0 or REAL_FPS is None:
        REAL_FPS = 30

    print(f"🎯 Using FPS: {REAL_FPS}")

    # -------- OUTPUT --------
    save_path = "output"
    os.makedirs(save_path, exist_ok=True)

    writers = []
    for i, cap in enumerate(caps):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        out = cv2.VideoWriter(
            f"{save_path}/lane{i+1}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        writers.append(out)

    while True:

        frames = []
        lane_counts = []
        all_finished = True
        emergency_lane = -1

        for cam_id, cap in enumerate(caps):

            if finished[cam_id]:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                lane_counts.append(0)
                continue

            ret, frame = cap.read()

            if not ret:
                print(f"✅ Video finished: {sources[cam_id]}")
                finished[cam_id] = True
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                lane_counts.append(0)
                continue
            else:
                all_finished = False

            im0 = frame.copy()

            # -------- PREPROCESS --------
            img = cv2.resize(im0, (imgsz[0], imgsz[1]))
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float() / 255.0
            img = img.unsqueeze(0)

            # -------- INFERENCE --------
            pred = model(img)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            count = 0

            for det in pred:
                if len(det):

                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape)

                    for *xyxy, conf, cls in det:

                        x1, y1, x2, y2 = map(int, xyxy)

                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        current_center = np.array([cx, cy])
                        matched_id = None

                        # -------- TRACKING --------
                        for vid, prev_center in vehicle_tracks.items():
                            distance = np.linalg.norm(current_center - prev_center)
                            if distance < max_tracking_distance:
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

                            # -------- FIXED SPEED --------
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

                        # -------- DRAW --------
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(im0, f"ID:{matched_id} {speed}km/h",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 255), 2)

            lane_counts.append(count)
            frames.append(im0)

        if all_finished:
            print("🎉 All videos completed")
            break

        # -------- SIGNAL LOGIC --------
        priority_scores = []
        for i in range(len(lane_counts)):
            priority = (lane_counts[i] * 2) + (lane_wait_time[i] * 1)
            priority_scores.append(priority)

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
    parser.add_argument("--conf-thres", type=float, default=0.25)
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