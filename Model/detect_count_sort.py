import argparse
import cv2
import torch
from sort import Sort

@torch.no_grad()
def run(weights, source, conf_thres):
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                            path=weights, force_reload=False)

    model.conf = conf_thres
    tracker = Sort()
    counted_ids = set()
    total_count = 0

    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append([x1, y1, x2, y2, conf])

        tracks = tracker.update(detections)

        line_y = frame.shape[0] // 2  # Adjust as needed
        
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw line
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255,0,0), 3)

            # Count only if vehicle crosses line
            if track_id not in counted_ids and cy > line_y:
                counted_ids.add(track_id)
                total_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

        cv2.putText(frame, f"Total Vehicles: {total_count}",
                    (30,60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0,0,255), 4)

        cv2.imshow("Traffic Count", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--conf-thres', type=float, default=0.6)
    opt = parser.parse_args()

    run(opt.weights, opt.source, opt.conf_thres)
