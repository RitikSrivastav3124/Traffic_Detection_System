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
#     finished = [False] * len(sources)

#     for src in sources:
#         cap = cv2.VideoCapture(src)
#         if not cap.isOpened():
#             print(f"❌ Cannot open {src}")
#         else:
#             print(f"✅ Loaded {src}")
#         caps.append(cap)

#     # -------- OUTPUT SETUP --------
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

#         if all_finished:
#             print("🎉 All videos completed")
#             break

#         # -------- SIGNAL LOGIC --------
#         max_lane = np.argmax(lane_counts)
#         signals = ["RED"] * len(frames)
#         signals[max_lane] = "GREEN"

#         # -------- DISPLAY + SAVE --------
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

#             writers[i].write(frame)

#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     # -------- CLEANUP --------
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

