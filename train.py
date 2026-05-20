import os
import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet512"
DETECTOR = "opencv"

def download_model():
    print("Downloading and setting up face recognition model...")
    
    # Create a dummy image to trigger model download
    import cv2
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite("dummy.jpg", dummy)
    
    try:
        DeepFace.represent(
            img_path="dummy.jpg",
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )
    except:
        pass
    
    # Save model config
    os.makedirs("saved_model", exist_ok=True)
    with open("saved_model/config.txt", "w") as f:
        f.write("model_name=Facenet512\n")
        f.write("detector=opencv\n")
        f.write("threshold=0.40\n")
    
    # Cleanup dummy image
    if os.path.exists("dummy.jpg"):
        os.remove("dummy.jpg")
    
    print("Model is ready!")

if __name__ == "__main__":
    download_model()