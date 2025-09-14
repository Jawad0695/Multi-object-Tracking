import cv2
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# ------------------------------
# Tracker Factory
# ------------------------------
def create_tracker(tracker_type="csrt"):
    if tracker_type == "csrt":
        return cv2.legacy.TrackerCSRT_create() if hasattr(cv2, "legacy") else cv2.TrackerCSRT_create()
    elif tracker_type == "kcf":
        return cv2.legacy.TrackerKCF_create() if hasattr(cv2, "legacy") else cv2.TrackerKCF_create()
    elif tracker_type == "mil":
        return cv2.legacy.TrackerMIL_create() if hasattr(cv2, "legacy") else cv2.TrackerMIL_create()
    elif tracker_type == "mosse":
        return cv2.legacy.TrackerMOSSE_create() if hasattr(cv2, "legacy") else cv2.TrackerMOSSE_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

# ------------------------------
# Helper Functions
# ------------------------------
def get_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def calculate_cost_matrix(trackers, detections):
    """Calculate cost matrix between trackers and detections using IoU"""
    cost_matrix = np.zeros((len(trackers), len(detections)))
    for i, (tracker_id, tracker_data) in enumerate(trackers.items()):
        box = tracker_data[1]  # Get the box (second element)
        for j, detection in enumerate(detections):
            cost_matrix[i, j] = 1 - get_iou(box, detection)  # Cost is 1 - IoU
    return cost_matrix

# ------------------------------
# Kalman Filter for Tracking
# ------------------------------
def create_kalman(x, y, w, h):
    kf = cv2.KalmanFilter(8, 4)  # state: x,y,w,h + velocities, measurement: x,y,w,h
    kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
    kf.transitionMatrix = np.eye(8, 8, dtype=np.float32)
    for i in range(4):
        kf.transitionMatrix[i, i+4] = 1.0
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
    kf.statePre[:4, 0] = np.array([x, y, w, h], dtype=np.float32)
    return kf

# ------------------------------
# Load YOLOv8 Model
# ------------------------------
# Load a pretrained YOLOv8 model (recommended for better performance)
model = YOLO("yolov8n.pt")  # Using nano version for speed, you can use "yolov8s.pt" for better accuracy

# ------------------------------
# Video + trackers
# ------------------------------
video_path = "/home/jawad/ws/src/multi-object-tracker/examples/assets/Cars Moving On Road Stock Footage - Free Download.mp4"
cap = cv2.VideoCapture(video_path)

trackers = {}
next_tracker_id = 0
frame_id = 0

# Parameters
detection_interval = 15
min_tracker_confidence = 3
max_failures = 8
min_iou_for_association = 0.4
max_age_without_detection = 30  # Frames a tracker can exist without detection
min_box_size = 30  # Minimum box size to consider for tracking

colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (128, 128, 0), (0, 128, 128),
          (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 0, 0)]

# For FPS calculation
prev_time = 0
fps = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_id += 1
    frame = cv2.resize(frame, (800, 500))
    height, width = frame.shape[:2]

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    current_detections = []
    detection_confidences = []

    # ----------------- Run YOLOv8 Detection -----------------
    if frame_id % detection_interval == 0:
        # Run YOLOv8 inference
        results = model(frame, conf=0.5, verbose=False)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Check if it's a car
                if model.names[class_id] == "car":
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Ensure the box is within frame boundaries and meets minimum size
                    x1 = max(0, min(x1, width - 5))
                    y1 = max(0, min(y1, height - 5))
                    w = max(min_box_size, min(w, width - x1))
                    h = max(min_box_size, min(h, height - y1))
                    
                    # Only add if box is large enough
                    if w >= min_box_size and h >= min_box_size:
                        current_detections.append((x1, y1, w, h))
                        detection_confidences.append(confidence)

    # ----------------- Update Trackers -----------------
    trackers_to_remove = []
    for tracker_id, tracker_data in list(trackers.items()):
        tracker, box, failures, successes, color, kf, age = tracker_data

        # Increment age
        age += 1
        
        # Kalman prediction
        prediction = kf.predict()
        pred_box = prediction[:4].flatten().astype(int)
        (px, py, pw, ph) = pred_box

        # Tracker update
        success, new_box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in new_box]
            measurement = np.array([[np.float32(x)], [np.float32(y)],
                                    [np.float32(w)], [np.float32(h)]])
            kf.correct(measurement)
            trackers[tracker_id] = (tracker, (x, y, w, h), 0, successes + 1, color, kf, 0)  # Reset age
            
            if successes >= min_tracker_confidence:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID: {tracker_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            failures += 1
            if failures > max_failures or age > max_age_without_detection:
                trackers_to_remove.append(tracker_id)
            else:
                # Use Kalman prediction when tracker fails
                trackers[tracker_id] = (tracker, (px, py, pw, ph), failures, successes, color, kf, age)
                if successes >= min_tracker_confidence:
                    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (200, 200, 200), 2)
                    cv2.putText(frame, f"ID: {tracker_id} (Pred)", (px, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    for tracker_id in trackers_to_remove:
        del trackers[tracker_id]

    # ----------------- Associate Detections with Hungarian Algorithm -----------------
    if frame_id % detection_interval == 0 and current_detections and trackers:
        # Create cost matrix
        cost_matrix = calculate_cost_matrix(trackers, current_detections)
        
        # Apply Hungarian algorithm
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Process assignments
            used_detections = set()
            for i, j in zip(row_ind, col_ind):
                tracker_id = list(trackers.keys())[i]
                tracker_data = trackers[tracker_id]
                _, _, _, successes, color, _, _ = tracker_data
                
                if cost_matrix[i, j] < (1 - min_iou_for_association):  # IoU > threshold
                    x, y, w, h = current_detections[j]
                    new_tracker = create_tracker("csrt")
                    new_tracker.init(frame, (x, y, w, h))
                    new_kf = create_kalman(x, y, w, h)
                    trackers[tracker_id] = (new_tracker, (x, y, w, h), 0, successes + 1, color, new_kf, 0)
                    used_detections.add(j)
            
            # Create new trackers for unmatched detections
            for j, detection in enumerate(current_detections):
                if j not in used_detections:
                    x, y, w, h = detection
                    
                    # Check if this detection overlaps significantly with any existing tracker
                    overlaps = False
                    for tracker_id, tracker_data in trackers.items():
                        existing_box = tracker_data[1]
                        iou = get_iou(existing_box, detection)
                        if iou > min_iou_for_association:
                            overlaps = True
                            break
                    
                    # Only create a new tracker if it doesn't overlap with existing ones
                    if not overlaps:
                        new_tracker = create_tracker("csrt")
                        new_tracker.init(frame, (x, y, w, h))
                        new_kf = create_kalman(x, y, w, h)
                        color = colors[next_tracker_id % len(colors)]
                        trackers[next_tracker_id] = (new_tracker, (x, y, w, h), 0, 1, color, new_kf, 0)
                        next_tracker_id += 1
                    
        except Exception as e:
            print(f"Error in Hungarian algorithm: {e}")
            # Fallback to simple association if Hungarian algorithm fails
            used_detections = set()
            for tracker_id, tracker_data in list(trackers.items()):
                box = tracker_data[1]  # Get the box
                _, _, successes, color, _, _, _ = tracker_data  # Get other needed values
                best_iou, best_idx = 0, -1
                for i, detection in enumerate(current_detections):
                    if i in used_detections:
                        continue
                    iou = get_iou(box, detection)
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
                if best_idx >= 0 and best_iou > min_iou_for_association:
                    x, y, w, h = current_detections[best_idx]
                    new_tracker = create_tracker("csrt")
                    new_tracker.init(frame, (x, y, w, h))
                    new_kf = create_kalman(x, y, w, h)
                    trackers[tracker_id] = (new_tracker, (x, y, w, h), 0, successes + 1, color, new_kf, 0)
                    used_detections.add(best_idx)

            # New trackers for unmatched detections
            for i, detection in enumerate(current_detections):
                if i not in used_detections:
                    x, y, w, h = detection
                    
                    # Check if this detection overlaps significantly with any existing tracker
                    overlaps = False
                    for tracker_id, tracker_data in trackers.items():
                        existing_box = tracker_data[1]
                        iou = get_iou(existing_box, detection)
                        if iou > min_iou_for_association:
                            overlaps = True
                            break
                    
                    # Only create a new tracker if it doesn't overlap with existing ones
                    if not overlaps:
                        new_tracker = create_tracker("csrt")
                        new_tracker.init(frame, (x, y, w, h))
                        new_kf = create_kalman(x, y, w, h)
                        color = colors[next_tracker_id % len(colors)]
                        trackers[next_tracker_id] = (new_tracker, (x, y, w, h), 0, 1, color, new_kf, 0)
                        next_tracker_id += 1
    elif frame_id % detection_interval == 0 and current_detections and not trackers:
        # Initialize trackers if none exist
        for i, detection in enumerate(current_detections):
            x, y, w, h = detection
            new_tracker = create_tracker("csrt")
            new_tracker.init(frame, (x, y, w, h))
            new_kf = create_kalman(x, y, w, h)
            color = colors[next_tracker_id % len(colors)]
            trackers[next_tracker_id] = (new_tracker, (x, y, w, h), 0, 1, color, new_kf, 0)
            next_tracker_id += 1

    # ----------------- Display -----------------
    active_trackers = sum(1 for tracker_data in trackers.values() if tracker_data[3] >= min_tracker_confidence)
    
    # Display information
    cv2.putText(frame, f"Frame: {frame_id}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Tracked Cars: {active_trackers}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Objects: {len(trackers)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Car Detection + Tracking + Kalman", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    time.sleep(0.02)

cap.release()
cv2.destroyAllWindows()
