cd /d %~dp0
call venv\Scripts\activate.bat 
call python tradeworker.py
cmd.exe