import os

for folder in ["data/frames/normal", "data/frames/medium"]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    print(f"{folder}: {len(files)} frames found")
