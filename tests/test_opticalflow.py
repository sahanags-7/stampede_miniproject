# tests/test_opticalflow.py
import cv2
import sys
import numpy as np
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.detection import detect_people  # HOG+SVM detection
from ultralytics import YOLO

SMOOTHING_FRAMES = 5
FRAME_SKIP = 2
RESIZE_DIM = (640, 360)

def process_frame(prev_frame, frame, motion_only=False, people_buffer=None, method="hog", yolo_model=None):
    frame = cv2.resize(frame, RESIZE_DIM)
    prev_frame = cv2.resize(prev_frame, RESIZE_DIM)
    display_frame = frame.copy()

    avg_people = 0
    if not motion_only and people_buffer is not None:
        # People detection (images / webcam)
        count = 0
        if method == "hog":
            count = detect_people(frame, method="hog")
        elif method == "yolo" and yolo_model:
            results = yolo_model(frame)
            if results:
                result = results[0]
                class_names = result.names
                person_class_index = next((k for k, v in class_names.items() if v == "person"), -1)
                if person_class_index != -1:
                    count = (result.boxes.cls == person_class_index).sum().item()

        # Update buffer
        people_buffer.append(count)
        if len(people_buffer) > SMOOTHING_FRAMES:
            people_buffer.pop(0)
        avg_people = int(np.mean(people_buffer))

    # Optical flow
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_level = np.mean(magnitude)

    # Risk assessment based on motion only for webcam
    if motion_only:
        if motion_level > 7.0:
            risk = "High"
            color = (0, 0, 255)
        elif motion_level > 3.0:
            risk = "Medium"
            color = (0, 165, 255)
        else:
            risk = "Low"
            color = (0, 255, 0)
    else:
        # Risk assessment for images/videos (people count and motion)
        if motion_level > 7.0 or avg_people > 20:
            risk = "High"
            color = (0, 0, 255)
        elif motion_level > 3.0 or avg_people > 8:
            risk = "Medium"
            color = (0, 165, 255)
        else:
            risk = "Low"
            color = (0, 255, 0)

    # Display text
    text = f"Motion: {motion_level:.2f}, Risk: {risk}"
    if not motion_only:
        text = f"People: {avg_people}, " + text

    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame, display_frame

def process_video(video_path, method="hog", yolo_model=None):
    print(f"▶ Processing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    ret, prev_frame = cap.read()
    if not ret:
        print(f"⚠️ Could not read video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)
    frame_count = 0

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        prev_frame, display_frame = process_frame(prev_frame, frame, motion_only=True)
        cv2.imshow("Crowd Risk Detection", display_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_webcam(method="hog", yolo_model=None):
    print("▶ Processing webcam feed")
    cap = cv2.VideoCapture(0)
    people_buffer = []

    ret, prev_frame = cap.read()
    if not ret:
        print("⚠️ Could not open webcam")
        return

    frame_count = 0
    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        prev_frame, display_frame = process_frame(prev_frame, frame, motion_only=False, people_buffer=people_buffer,
                                                  method=method, yolo_model=yolo_model)
        cv2.imshow("Live Crowd Risk Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path, method="hog", yolo_model=None):
    print(f"▶ Processing image: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"⚠️ Could not read image: {image_path}")
        return

    display_frame = cv2.resize(frame, RESIZE_DIM).copy()
    people_buffer = []
    _, display_frame = process_frame(display_frame, display_frame, motion_only=False, people_buffer=people_buffer,
                                     method=method, yolo_model=yolo_model)
    cv2.imshow("Image Crowd Risk Detection", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of dataset")
    parser.add_argument("--image", type=str, help="Path to a single image to process")
    parser.add_argument("--method", type=str, default="hog", choices=["hog", "yolo"], help="Detection method")
    args = parser.parse_args()

    yolo_model = None
    if args.method == "yolo":
        yolo_model = YOLO("yolov8n.pt")

    if args.image:
        process_image(args.image, method=args.method, yolo_model=yolo_model)
    elif args.webcam:
        process_webcam(method=args.method, yolo_model=yolo_model)
    else:
        dataset_folder = Path("data/raw")
        for img_path in sorted(dataset_folder.glob("*.png")) + sorted(dataset_folder.glob("*.jpg")):
            process_image(img_path, method=args.method, yolo_model=yolo_model)
        for vid_path in sorted(dataset_folder.glob("*.mp4")):
            process_video(vid_path, method=args.method, yolo_model=yolo_model)
