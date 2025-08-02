from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from io import BytesIO
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt")
model.to(device)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    results = model.predict(
        image_np,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        augment=False
    )

    detections = []
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

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

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Draw label and confidence
            text = f"{label} {conf:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
            draw.text((x1, y1 - text_height), text, fill="white", font=font)

    # Convert image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"detections": detections, "annotated_image": img_str}
