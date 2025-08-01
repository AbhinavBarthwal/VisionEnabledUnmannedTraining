@echo off
echo Running Industrial Safety Equipment Detection Model
echo =================================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check for trained model
if not exist "runs\train\industrial_safety_detection\weights\best.pt" (
    echo No trained model found. Starting training...
    python train_model.py --epochs 50 --batch-size 16
) else (
    echo Found trained model. Running inference demo...
    python demo.py --model runs\train\industrial_safety_detection\weights\best.pt
)

pause
