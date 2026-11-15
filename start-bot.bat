@echo off
setlocal enabledelayedexpansion
title Football Predictions Bot
cd /d "%~dp0"

echo ============================================
echo   Football Predictions Bot - launcher
echo ============================================

rem ---- 1. Node.js check ----
where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js is not installed. Download it from https://nodejs.org ^(LTS^), then run this again.
  pause
  exit /b 1
)

rem ---- 2. .env check ----
if not exist ".env" (
  if exist ".env.example" (
    copy ".env.example" ".env" >nul
    echo [INFO] Created .env from .env.example. Fill in your keys, save, and close Notepad.
    notepad ".env"
  ) else (
    echo [ERROR] .env file missing. Create it with BOT_TOKEN and FOOTBALL_DATA_KEY.
    pause
    exit /b 1
  )
)

rem ---- 3. Install dependencies once ----
if not exist "node_modules" (
  echo [INFO] Installing dependencies...
  call npm install
  if errorlevel 1 (
    echo [ERROR] npm install failed. Check your internet connection.
    pause
    exit /b 1
  )
)

rem ---- 4. Ollama (only if AI_PROVIDER=ollama in .env) ----
findstr /b /i /c:"AI_PROVIDER=ollama" ".env" >nul 2>nul
if errorlevel 1 goto :run

set "MODEL=qwen2.5:3b"
for /f "usebackq tokens=1,* delims==" %%a in (`findstr /b /i /c:"AI_MODELS=" ".env"`) do set "MODEL=%%b"
for /f "tokens=1 delims=, " %%b in ("!MODEL!") do set "MODEL=%%b"
if "!MODEL!"=="" set "MODEL=qwen2.5:3b"

where ollama >nul 2>nul
if errorlevel 1 (
  echo [WARN] Ollama not found. Install it from https://ollama.com/download - the bot will run without AI for now.
  goto :run
)

ollama list >nul 2>nul
if errorlevel 1 (
  echo [INFO] Starting Ollama...
  start "" /min ollama serve
  timeout /t 6 /nobreak >nul
)

ollama list > "%temp%\ollama_models.txt" 2>nul
findstr /i /c:"!MODEL!" "%temp%\ollama_models.txt" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Downloading AI model !MODEL! ^(one-time, about 2 GB^)...
  ollama pull !MODEL!
)
echo [OK] Ollama model: !MODEL!

rem ---- 5. Run the bot, restart automatically if it crashes ----
:run
:loop
echo.
echo [%date% %time%] Starting bot... ^(close this window or press Ctrl+C to stop^)
node index.js
echo.
echo [%date% %time%] Bot stopped. Restarting in 10 seconds...
timeout /t 10 /nobreak >nul
goto loop