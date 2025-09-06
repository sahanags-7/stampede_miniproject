import cv2
import numpy as np

def test_environment():
    print("✅ Python environment is ready!")
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)

if __name__ == "__main__":
    test_environment()
