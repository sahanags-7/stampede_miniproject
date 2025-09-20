import os, shutil

src_dirs = ["data/frames/abnormal1", "data/frames/abnormal2"]
dst_dir = "data/frames/abnormal_combined"
os.makedirs(dst_dir, exist_ok=True)

counter = 0
for src in src_dirs:
    for f in sorted(os.listdir(src)):
        if f.lower().endswith(".jpg"):
            shutil.copy(os.path.join(src, f), os.path.join(dst_dir, f"frame_{counter}.jpg"))
            counter += 1
print(f"✅ Combined {counter} frames into {dst_dir}")
