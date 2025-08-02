@echo off
echo Setting up Industrial Safety Equipment Detection Model
echo =====================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or later
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

echo.
echo Setup complete!
echo.
echo To activate the environment manually, run:
echo   venv\Scripts\activate.bat
echo.
echo To start training, run:
echo   python train_model.py
echo.
pause
