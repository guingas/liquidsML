@echo off
title Dispositivo S - Instalacao
echo ============================================================
echo   DISPOSITIVO S - Setup Automatico
echo ============================================================
echo.

cd /d "%~dp0"

:: Verificar se Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado!
    echo Instale Python 3.10+ de: https://www.python.org/downloads/
    echo Marque "Add Python to PATH" durante a instalacao.
    pause
    exit /b 1
)

echo [1/3] Criando ambiente virtual...
if not exist .venv (
    python -m venv .venv
    echo       Ambiente virtual criado.
) else (
    echo       Ambiente virtual ja existe.
)

echo [2/3] Ativando ambiente virtual...
call .venv\Scripts\activate.bat

echo [3/3] Instalando dependencias (pode demorar na primeira vez)...
pip install -r requirements.txt --quiet

echo.
echo ============================================================
echo   Instalacao concluida!
echo   Para rodar o webapp, execute: run_webapp.bat
echo ============================================================
pause
