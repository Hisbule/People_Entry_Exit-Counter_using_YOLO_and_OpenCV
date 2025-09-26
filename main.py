# Library
import cv2
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
from ultralytics import YOLO

#  Configuration
video_path = r"C:\Users\uddin\Downloads\people-walking.mp4"  # Input video
frame_width, frame_height = 1280, 720
upper_line_y = int(frame_height * 0.3)
lower_line_y = int(frame_height * 0.7)
max_track_distance = 50

#  Counters and Storage 
in_count = 0
out_count = 0
next_person_id = 0
centroids = {}
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
motion_history = deque(maxlen=50)

# Load Video and YOLOv5 Model 
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov5s.pt')  # Automatically downloads pre-trained model

#  Setup Video Writer 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_people_counter.avi', fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    #  YOLOv5 Detection 
    results = model(frame, verbose=False)[0]
    new_centroids = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] != 'person':
            continue  # Only detect persons

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cX, cY = int((x1 + x2) / 2), int((y1 + y2) / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        matched = False
        for pid, (px, py) in centroids.items():
            if dist.euclidean((cX, cY), (px, py)) < max_track_distance:
                new_centroids[pid] = (cX, cY)

                if py < upper_line_y and cY >= upper_line_y:
                    in_count += 1
                elif py > lower_line_y and cY <= lower_line_y:
                    out_count += 1

                matched = True
                break

        if not matched:
            new_centroids[next_person_id] = (cX, cY)
            next_person_id += 1

    centroids = new_centroids

    # === Draw Tracking and Lines ===
    for pid, (cX, cY) in centroids.items():
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID: {pid}", (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.line(frame, (0, upper_line_y), (frame_width, upper_line_y), (0, 0, 255), 2)
    cv2.line(frame, (0, lower_line_y), (frame_width, lower_line_y), (255, 0, 0), 2)

    cv2.putText(frame, f"IN: {in_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"OUT: {out_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    #  Save & Show Frame 
    out.write(frame)
    cv2.imshow("YOLO People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  Cleanup 
cap.release()
out.release()
cv2.destroyAllWindows()
