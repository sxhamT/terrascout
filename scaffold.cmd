@echo off
REM TerraScout repo scaffold script
REM Run from plain cmd (no admin) once, from any directory.
REM Creates D:\project\terrascout and all subdirectories.
REM Copies identical files from dronos, then creates stubs for new files.

set DRONOS=D:\project\dronos
set TS=D:\project\terrascout

echo.
echo === TerraScout scaffold starting ===
echo   Source:  %DRONOS%
echo   Target:  %TS%
echo.

REM ── Create directory tree ────────────────────────────────────────────────
mkdir "%TS%"
mkdir "%TS%\config"
mkdir "%TS%\terrain"
mkdir "%TS%\scanner"
mkdir "%TS%\controller"
mkdir "%TS%\docs"

REM ── Copy identical files from dronos ─────────────────────────────────────
echo Copying reusable files from dronos...

copy "%DRONOS%\controller\pid_controller.py"  "%TS%\controller\pid_controller.py"
copy "%DRONOS%\controller\observe.py"          "%TS%\controller\observe.py"
copy "%DRONOS%\controller\act.py"              "%TS%\controller\act.py"
copy "%DRONOS%\launch_ros2.cmd"                "%TS%\launch_ros2.cmd"
copy "%DRONOS%\.gitignore"                     "%TS%\.gitignore"
copy "%DRONOS%\.gitattributes"                 "%TS%\.gitattributes"

REM ── Patch launch_ros2.cmd — replace dronos paths with terrascout paths ───
echo Patching launch_ros2.cmd paths...
powershell -Command "(Get-Content '%TS%\launch_ros2.cmd') -replace 'dronos','terrascout' -replace 'launch_gui\.py','terrascout_main.py' | Set-Content '%TS%\launch_ros2.cmd'"

REM ── Create __init__.py stubs ──────────────────────────────────────────────
echo. > "%TS%\terrain\__init__.py"
echo. > "%TS%\scanner\__init__.py"
echo. > "%TS%\controller\__init__.py"

REM ── Git init ──────────────────────────────────────────────────────────────
echo Initialising git repo...
cd /d "%TS%"
git init
git checkout -b main

echo.
echo === Scaffold complete ===
echo Next: copy CLAUDE.md + source stubs into %TS%, then:
echo   git add .
echo   git commit -m "feat: initial scaffold"
echo.
