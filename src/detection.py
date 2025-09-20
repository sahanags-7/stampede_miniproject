import cv2
import os
import json

def detect_people(image_path):
    """Detect people in a single image using HOG + SVM."""
    # Initialize HOG + SVM
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return 0

    boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
    return len(boxes)

def process_frames(input_dir, output_json, label, step=10):
    """Process all frames in a directory and save results to JSON."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    results = []
    files = sorted(os.listdir(input_dir))
    
    if not files:
        print(f"⚠️ No files found in {input_dir}")
        return

    for img_file in files[::step]:  # Process every nth frame
        if img_file.lower().endswith((".jpg", ".png")):
            print(f"Processing: {img_file}")  # Debug print
            path = os.path.join(input_dir, img_file)
            count = detect_people(path)

            # Safe frame ID parsing
            try:
                frame_id = int(os.path.splitext(img_file)[0].split("_")[-1])
            except:
                frame_id = len(results) + 1

            results.append({
                "frame_id": frame_id,
                "people_count": count,
                "density_status": label
            })
    
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Results saved to {output_json}")

if __name__ == "__main__":
    # Process normal frames
    process_frames("data/frames/normal", "results/normal_counts.json", "normal", step=5)

    # Process medium frames
    process_frames("data/frames/medium", "results/medium_counts.json", "medium", step=5)

    # Process combined abnormal frames
    process_frames("data/frames/abnormal_combined", "results/abnormal_counts.json", "abnormal", step=5)

