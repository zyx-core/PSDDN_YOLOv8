# PSDDN Local Setup Script for Windows

Write-Host "--- Starting PSDDN Local Setup ---" -ForegroundColor Cyan

# 1. Create Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# 2. Activate and Install Dependencies
Write-Host "Installing dependencies (this may take a few minutes)..."
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements_local.txt

# 3. Create necessary folders
Write-Host "Creating data and result directories..."
New-Item -ItemType Directory -Force -Path "data\shanghaitech\labels\train"
New-Item -ItemType Directory -Force -Path "data\shanghaitech\folds"
New-Item -ItemType Directory -Force -Path "runs"

Write-Host "`n✅ Setup Complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: .\venv\Scripts\Activate.ps1"
