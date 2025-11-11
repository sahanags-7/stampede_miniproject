import cv2, os

def extract_frames(video_path, output_dir, resize_dim=(640,480)):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        cv2.imwrite(f"{output_dir}/frame_{frame_id}.jpg", frame)
        frame_id += 1
    
    cap.release()
    print(f"✅ Extracted {frame_id} frames to {output_dir}")

if __name__ == "__main__":
    extract_frames("data/raw/normal_crowd.mp4", "data/frames/normal")
    extract_frames("data/raw/medium_crowd.mp4", "data/frames/medium")
    extract_frames("data/raw/abnormal_crowd1.mp4", "data/frames/abnormal1")
    extract_frames("data/raw/abnormal_crowd2.mp4", "data/frames/abnormal2")
