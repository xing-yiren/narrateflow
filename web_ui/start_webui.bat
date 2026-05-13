@echo off
setlocal
cd /d D:\qwen3-tts
set "PATH=D:\qwen3-tts\tools\sox\sox-14.4.2;%PATH%"
call venv313\Scripts\activate.bat
python webui.py
