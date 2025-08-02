from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

# Allow frontend (adjust URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with http://localhost:5173 or vercel.app in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = YOLO("best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model = YOLO("best.pt").to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    results = model.predict(image_np, conf=0.25, iou=0.4, imgsz=640)
    print("✅ Detections:", results[0].boxes)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1]
            })

    return {"detections": detections}
