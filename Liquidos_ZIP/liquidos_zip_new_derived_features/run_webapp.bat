@echo off
title Dispositivo S - Web App
cd /d "%~dp0"

:: Se .venv nao existe, rodar setup primeiro
if not exist .venv (
    echo Primeira execucao detectada. Instalando...
    call setup.bat
)

call .venv\Scripts\activate.bat
echo Abrindo Dispositivo S no navegador...
streamlit run webapp.py
pause
