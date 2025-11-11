import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 once
yolo_model = YOLO("yolov8n.pt")  # pretrained small model

def detect_people(input_data, method="yolo"):
    """
    Detect people using HOG+SVM (default) or YOLOv8.
    method: "hog" or "yolo"
    """
    # Handle both file paths and frames
    img = input_data if isinstance(input_data, np.ndarray) else cv2.imread(input_data)
    if img is None:
        print(f"⚠️ Could not read image: {input_data}")
        return 0

    if method == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(img, winStride=(8, 8))
        return len(boxes)

    elif method == "yolo":
        results = yolo_model.predict(img, verbose=False)
        people = 0
        for r in results:
            for cls in r.boxes.cls:
                if int(cls) == 0:  # class 0 = 'person'
                    people += 1
        return people
