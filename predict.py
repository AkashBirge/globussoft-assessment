import os
import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet512"
DETECTOR = "opencv"
THRESHOLD = 0.40

def load_model():
    print(f"Using model: {MODEL_NAME}")
    return MODEL_NAME

def get_embedding_and_bbox(image_path):
    results = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True
    )
    best = results[0]
    embedding = np.array(best["embedding"])
    area = best.get("facial_area", {})
    bbox = {"x": area.get("x", 0), "y": area.get("y", 0),
            "w": area.get("w", 0), "h": area.get("h", 0)}
    return embedding, bbox

def predict(image_path_1, image_path_2):
    emb1, bbox1 = get_embedding_and_bbox(image_path_1)
    emb2, bbox2 = get_embedding_and_bbox(image_path_2)

    # Calculate similarity
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    similarity = float(dot / norm) if norm != 0 else 0.0

    # Decision
    is_same = similarity >= (1.0 - THRESHOLD)
    verdict = "same person" if is_same else "different person"

    return {
        "verification_result": verdict,
        "similarity_score": round(similarity, 4),
        "bbox_image1": bbox1,
        "bbox_image2": bbox2
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict.py image1.jpg image2.jpg")
    else:
        load_model()
        result = predict(sys.argv[1], sys.argv[2])
        print("Result:", result["verification_result"])
        print("Similarity:", result["similarity_score"])
        print("BBox1:", result["bbox_image1"])
        print("BBox2:", result["bbox_image2"])