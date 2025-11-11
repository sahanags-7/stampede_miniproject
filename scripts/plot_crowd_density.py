import json
import matplotlib.pyplot as plt

# Load JSON data
def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)

normal_data = load_json("results/normal_counts.json")
medium_data = load_json("results/medium_counts.json")
abnormal_data = load_json("results/abnormal_counts.json")

# Extract frame IDs and people counts
def extract_counts(data):
    frame_ids = [d["frame_id"] for d in data]
    counts = [d["people_count"] for d in data]
    return frame_ids, counts

normal_frames, normal_counts = extract_counts(normal_data)
medium_frames, medium_counts = extract_counts(medium_data)
abnormal_frames, abnormal_counts = extract_counts(abnormal_data)

# Plot all three categories
plt.figure(figsize=(12, 6))
plt.plot(normal_frames, normal_counts, label="Normal", color="green")
plt.plot(medium_frames, medium_counts, label="Medium", color="orange")
plt.plot(abnormal_frames, abnormal_counts, label="Abnormal", color="red")
plt.xlabel("Frame ID")
plt.ylabel("People Count")
plt.title("Crowd Density: Normal vs Medium vs Abnormal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
