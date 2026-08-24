@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"
set "PYTHONUTF8=1"

for %%V in (3.14 3.13 3.12 3.11) do (
    py -V:%%V -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 15) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_SELECTOR=-V:%%V"
        goto run_py
    )
)

for %%V in (3.14 3.13 3.12 3.11) do (
    py -%%V -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 15) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_SELECTOR=-%%V"
        goto run_py
    )
)

python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 15) else 1)" >nul 2>&1
if not errorlevel 1 goto run_python

echo 启动失败：未找到 Python 3.11.x 至 3.14.x。
echo 建议安装 Python 3.14，然后重新双击 start_windows.bat。
set "EXIT_CODE=1"
goto finish

:run_py
py %PYTHON_SELECTOR% "%~dp0scripts\launch_app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_python
python "%~dp0scripts\launch_app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

:finish
if "%EXIT_CODE%"=="0" exit /b 0
if "%EXIT_CODE%"=="130" exit /b 130
echo.
pause
exit /b %EXIT_CODE%
