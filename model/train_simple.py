from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')

print("🚀 Starting OPTIMIZED Training for High Precision & Recall")
print("🎯 Target: >90% Precision and >80% Recall")
print("=" * 60)

# Start optimized training
results = model.train(
    data=r'C:\Users\user\OneDrive\Documents\pro\sambhav\HackByte_Dataset\yolo_params.yaml',
    name='industrial_safety_optimized',
    epochs=80,
    batch=16,
    imgsz=640,
    device='cpu',
    workers=8,
    lr0=0.002,
    lrf=0.0001,
    momentum=0.95,
    weight_decay=0.001,
    optimizer='AdamW',
    cos_lr=True,
    warmup_epochs=5,
    box=8.0,
    cls=1.0,
    dfl=1.5,
    hsv_h=0.03,
    hsv_s=0.9,
    hsv_v=0.6,
    degrees=8.0,
    translate=0.2,
    scale=0.7,
    shear=3.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.4,
    val=True,
    plots=True,
    save=True,
    save_period=10,
    patience=20,
    close_mosaic=15,
    amp=False
)

print("✅ OPTIMIZED TRAINING COMPLETED!")
print(f"📈 Results: {results.save_dir}")
