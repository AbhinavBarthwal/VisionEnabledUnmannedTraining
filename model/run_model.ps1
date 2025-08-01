# PowerShell script to run Industrial Safety Equipment Detection Model
# =================================================================

Write-Host "Running Industrial Safety Equipment Detection Model" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Error: Virtual environment not found" -ForegroundColor Red
    Write-Host "Please run setup.bat first" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Check for trained model
$modelPath = "runs\train\industrial_safety_detection\weights\best.pt"
if (-not (Test-Path $modelPath)) {
    Write-Host "No trained model found. Starting training..." -ForegroundColor Yellow
    python train_model.py --epochs 50 --batch-size 16
} else {
    Write-Host "Found trained model. Running inference demo..." -ForegroundColor Yellow
    python demo.py --model $modelPath
}

Read-Host "Press Enter to exit"
