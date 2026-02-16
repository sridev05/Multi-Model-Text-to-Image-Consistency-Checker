# Text-to-Image Consistency Checker - Start Everything
# This PowerShell script starts both the Python backend and React frontend

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "Text-to-Image Consistency Checker" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "text_image_consistency")) {
    Write-Host "Error: Please run this script from the root directory" -ForegroundColor Red
    Write-Host "Expected to find: text_image_consistency folder" -ForegroundColor Red
    exit 1
}

Write-Host "Starting backend and frontend..." -ForegroundColor Yellow
Write-Host ""

# Start backend in a new PowerShell window
Write-Host "Starting Backend API..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd text_image_consistency; python src/api.py"

# Wait for backend to start
Start-Sleep -Seconds 3

# Start frontend in a new PowerShell window
Write-Host "Starting Frontend..."  -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd text_image_consistency\frontend; npm run dev"

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "Services are starting:" -ForegroundColor Yellow
Write-Host "  Backend API: http://localhost:5000" -ForegroundColor Cyan
Write-Host "  Frontend:    http://localhost:5173" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Once both are running, open http://localhost:5173 in your browser" -ForegroundColor Yellow
Write-Host "Press any key to exit this window..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
