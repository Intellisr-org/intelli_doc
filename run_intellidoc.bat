@echo off
echo ========================================
echo    IntelliDoc Flask Application
echo ========================================
echo.

:: Change to the intellidoc directory
cd /d "%~dp0intellidoc"

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and try again
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )

    :: Activate virtual environment
    echo Activating virtual environment...
    call venv\Scripts\activate.bat

    :: Install/upgrade pip
    echo Upgrading pip...
    python -m pip install --upgrade pip

    :: Install requirements
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if weights directory exists
if not exist "weights" (
    echo WARNING: weights directory not found
    echo Please ensure you have the required model weights in the weights/ directory
    echo.
)

:: Set environment variables for performance
set RECOGNITION_BATCH_SIZE=256
set DETECTOR_BATCH_SIZE=18
set ORDER_BATCH_SIZE=16
set RECOGNITION_STATIC_CACHE=true

:: Run the Flask application
echo.
echo Starting IntelliDoc Flask Application...
echo The application will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
python flask_app.py

:: Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    pause
)

:: Deactivate virtual environment
call venv\Scripts\deactivate.bat 