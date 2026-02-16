@echo off
REM Text-to-Image Consistency Checker - Start Everything
REM This script starts both the Python backend and React frontend

echo.
echo ===================================================
echo Text-to-Image Consistency Checker
echo ===================================================
echo.

REM Check if we're in the right directory
if not exist "text_image_consistency" (
    echo Error: Please run this script from the root directory
    echo Expected to find: text_image_consistency folder
    pause
    exit /b 1
)

echo Starting backend and frontend...
echo.

REM Open two terminals - one for backend, one for frontend
REM Backend
start "Backend API" cmd /k "cd text_image_consistency && python src/api.py"

REM Give backend time to start
timeout /t 3 /nobreak

REM Frontend
start "Frontend" cmd /k "cd text_image_consistency\frontend && npm run dev"

echo.
echo ===================================================
echo Services are starting:
echo   Backend API: http://localhost:5000
echo   Frontend:    http://localhost:5173
echo ===================================================
echo.
echo Once both are running, open http://localhost:5173 in your browser
echo Close the windows to stop the services
pause
