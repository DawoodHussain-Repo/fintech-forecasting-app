@echo off@echo off

echo ========================================echo ========================================

echo Starting FinTech Forecasting Frontendecho FinTech Forecasting App - Start Script

echo ========================================echo ========================================

echo.echo.



REM Start Frontend (Next.js)REM Get the current directory

echo Starting Next.js development server...set "APP_DIR=%~dp0"

npm run dev
echo Starting FinTech Forecasting Application...
echo.

REM Start Backend (Flask) in a new window
echo [1/2] Starting Backend Server...
start "FinTech Backend" cmd /k "cd /d "%APP_DIR%backend" && .\venv\Scripts\activate && echo Backend server starting at http://localhost:5000 && python run_server.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start Frontend (Next.js) in a new window
echo [2/2] Starting Frontend Server...
start "FinTech Frontend" cmd /k "cd /d "%APP_DIR%" && echo Frontend server starting at http://localhost:3000 && npm run dev"

echo.
echo ========================================
echo Application Started Successfully!
echo ========================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Both servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
echo Press any key to exit this launcher...
pause >nul