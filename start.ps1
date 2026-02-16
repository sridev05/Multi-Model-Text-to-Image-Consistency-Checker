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

# Step 1: Install dependencies
Write-Host "Step 1: Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take 2-5 minutes on first run..." -ForegroundColor Yellow
Push-Location "text_image_consistency"
pip install -r requirements.txt -q
Pop-Location

Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Starting services..." -ForegroundColor Yellow
Write-Host ""

# Start backend in a new PowerShell window
Write-Host "  → Starting Backend API on http://localhost:5000" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\text_image_consistency'; Write-Host 'Backend starting...' -ForegroundColor Green; python src/api.py"

# Wait for backend to start
Start-Sleep -Seconds 5

# Start frontend in a new PowerShell window
Write-Host "  → Starting Frontend on http://localhost:5173" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\text_image_consistency\frontend'; Write-Host 'Frontend starting...' -ForegroundColor Green; npm run dev"

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "✓ Services are starting!" -ForegroundColor Yellow
Write-Host "  Backend API: http://localhost:5000" -ForegroundColor Cyan
Write-Host "  Frontend:    http://localhost:5173" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT:" -ForegroundColor Yellow
Write-Host "  → Keep both terminal windows OPEN!" -ForegroundColor Yellow
Write-Host "  → Open http://localhost:5173 in your browser" -ForegroundColor Cyan
Write-Host "  → Wait 10-15 seconds for models to load on first run" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
