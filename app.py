import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from predict import load_model, predict

app = FastAPI(
    title="Face Authentication API",
    description="Upload two face images to check if they are the same person",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    load_model()
    print("Face auth model loaded and ready!")

@app.get("/")
def root():
    return {"status": "ok", "message": "Face Authentication API is running!"}

@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    tmp_dir = tempfile.mkdtemp()
    try:
        path1 = os.path.join(tmp_dir, "img1.jpg")
        path2 = os.path.join(tmp_dir, "img2.jpg")

        with open(path1, "wb") as f:
            shutil.copyfileobj(image1.file, f)
        with open(path2, "wb") as f:
            shutil.copyfileobj(image2.file, f)

        result = predict(path1, path2)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)