@echo off
rem ============================================================
rem  DS-Labeler launcher
rem  Python 解释器按以下顺序解析（找到即用）：
rem    1) 环境变量 DSLABELER_PY（最高优先）
rem    2) 本脚本同目录的 python_path.txt（一行：解释器完整路径）
rem    3) PATH 上的 python
rem ============================================================
setlocal
set "ROOT=%~dp0"
set "PY="

if defined DSLABELER_PY set "PY=%DSLABELER_PY%"
if not defined PY if exist "%ROOT%python_path.txt" set /p PY=<"%ROOT%python_path.txt"
if not defined PY set "PY=python"

rem 带路径的解释器直接看文件是否存在；裸名（python / python3）交给 where 解析
if exist "%PY%" goto found
for /f "delims=" %%i in ('where "%PY%" 2^>nul') do if not defined RESOLVED set "RESOLVED=%%i"
if defined RESOLVED set "PY=%RESOLVED%"
if not defined RESOLVED (
    echo [ERROR] Python not found: %PY%
    echo Set DSLABELER_PY, or put the interpreter path in python_path.txt,
    echo or make sure python is on PATH.
    pause
    exit /b 1
)

:found
cd /d "%ROOT%"
"%PY%" "%ROOT%app\main.py" %*
set "EXIT=%ERRORLEVEL%"
if not "%EXIT%"=="0" (
    echo [ERROR] DS-Labeler exited with code %EXIT%
    pause
)
exit /b %EXIT%
