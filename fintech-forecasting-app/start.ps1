# FinTech Forecasting App - PowerShell Start Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FinTech Forecasting App - Start Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$AppDir = Get-Location

Write-Host "Starting FinTech Forecasting Application..." -ForegroundColor Green
Write-Host ""

# Start Backend (Flask)
Write-Host "[1/2] Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$AppDir\backend'; .\venv\Scripts\Activate.ps1; Write-Host 'Backend server starting at http://localhost:5000' -ForegroundColor Green; python run_server.py"

# Wait for backend to start
Write-Host "Waiting for backend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# Start Frontend (Next.js)
Write-Host "[2/2] Starting Frontend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$AppDir'; Write-Host 'Frontend server starting at http://localhost:3000' -ForegroundColor Green; npm run dev"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Application Started Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backend:  http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Both servers are running in separate PowerShell windows." -ForegroundColor Gray
Write-Host "Close those windows to stop the servers." -ForegroundColor Gray
Write-Host ""
Write-Host "Opening frontend in browser..." -ForegroundColor Yellow

# Wait a moment then open browser
Start-Sleep -Seconds 5
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "Press any key to exit this launcher..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")