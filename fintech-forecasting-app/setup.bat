@echo off
REM FinTech Forecasting App Setup Script for Windows
REM This script sets up the development environment for the forecasting application

echo ========================================
echo FinTech Forecasting App Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    exit /b 1
)

REM Check if MongoDB is running (optional for MongoDB Atlas)
echo Checking MongoDB Atlas connection...
echo Using MongoDB Atlas - local MongoDB not required
echo.

echo Step 1: Setting up Frontend Dependencies
echo ----------------------------------------
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies
    exit /b 1
)

echo.
echo Step 2: Setting up Backend Environment
echo --------------------------------------
cd backend

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies (Windows compatible)...
pip install --upgrade pip

REM Try Windows-specific requirements first, fallback to regular requirements
if exist requirements-windows.txt (
    echo Using Windows-compatible requirements...
    pip install -r requirements-windows.txt
) else (
    echo Using standard requirements...
    pip install -r requirements.txt
)

if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    echo Try installing packages individually if needed
    exit /b 1
)

cd ..

echo.
echo Step 3: Environment Configuration
echo ---------------------------------
if not exist .env.local (
    echo Creating .env.local for frontend...
    echo ALPHA_VANTAGE_API_KEY=demo > .env.local
    echo NOTE: Update .env.local with your actual Alpha Vantage API key
)

if not exist backend\.env (
    echo Creating backend .env file...
    copy backend\.env.example backend\.env
    echo NOTE: Update backend\.env with your configuration
)

REM Create necessary directories
if not exist backend\logs mkdir backend\logs
if not exist backend\models mkdir backend\models

echo.
echo Step 4: Database Setup (MongoDB Atlas)
echo ---------------------------------
echo Using MongoDB Atlas cloud database
echo Make sure to replace <db_password> in .env files with your actual password
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application:
echo.
echo 1. Start the backend (in backend directory):
echo    venv\Scripts\activate
echo    python main.py
echo.
echo 2. Start the frontend (in root directory):
echo    npm run dev
echo.
echo 3. Or use Docker Compose:
echo    docker-compose up -d
echo.
echo Access the application at: http://localhost:3000
echo Backend API available at: http://localhost:5000
echo.
echo Don't forget to:
echo - Update API keys in .env files
echo - Replace <db_password> with your MongoDB Atlas password
echo - MongoDB Atlas and Redis services are managed externally
echo ========================================

pause
