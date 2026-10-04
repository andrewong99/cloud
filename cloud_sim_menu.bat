@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Copyright (C) 2026 The CloudSim authors
rem CloudSim is free software: you can redistribute it and/or modify it under the
rem terms of the GNU General Public License, version 3 or (at your option) any
rem later version.  It comes with NO WARRANTY; see the LICENSE file.
rem CloudSim menu: start the sky, or run the checks, without typing a command.
rem Double-click it.  Keep it next to cloud_sim.py.
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title CloudSim

rem The Python you type in the command line ("python"); else the launcher ("py -3").
set "PY=python"
python --version >nul 2>&1
if errorlevel 1 (
    py -3 --version >nul 2>&1
    if errorlevel 1 (
        echo Python was not found.  Install it from python.org and tick "Add python.exe to PATH".
        pause
        exit /b 1
    )
    set "PY=py -3"
)

rem The packages, the first time (as cloud_sim.bat does).
%PY% -c "import moderngl, pyglet, numpy, scipy, PIL" >nul 2>&1
if errorlevel 1 (
    echo CloudSim needs a few Python packages the first time you run it.
    echo Installing: moderngl pyglet numpy scipy pillow
    echo.
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo The install failed - the message is above.
        pause
        exit /b 1
    )
)

:menu
echo.
echo   CloudSim
echo   --------
echo   1  Start: live weather, Kuala Lumpur
echo   2  Start with choices: place, time, offline, cloud model, quality, window
echo   3  Self-test
echo   4  Quit
echo.
set "PICK="
set /p "PICK=  Choose 1-4 and press Enter [1]: "
if defined PICK set "PICK=%PICK:"=%"
if "%PICK%"=="" set "PICK=1"
if "%PICK%"=="1" goto start
if "%PICK%"=="2" goto choices
if "%PICK%"=="3" goto selftest
if "%PICK%"=="4" goto end
echo   %PICK% is not one of 1-4.
goto menu

:start
set "ARGS="
call :run
goto menu

:choices
set "ARGS="
echo.
echo   Press Enter to keep what is in brackets.
echo.
set "PLACE="
set /p "PLACE=  Place: a city, or latitude,longitude - e.g. Paris or 48.86,2.35 [Kuala Lumpur]: "
rem a place and a name are free text: no quotes or characters the command line would act on
if defined PLACE set "PLACE=%PLACE:"=%"
if defined PLACE set "PLACE=%PLACE:&=and%"
if defined PLACE set "PLACE=%PLACE:|=%"
if defined PLACE set "PLACE=%PLACE:<=%"
if defined PLACE set "PLACE=%PLACE:>=%"
if defined PLACE set "PLACE=%PLACE:^=%"
if "%PLACE%"=="" goto ask_time
rem --place="..." so a city of two words, or a latitude south (-33.87,...), stays one value
set "ARGS=--place="%PLACE%""
set "NAME="
set /p "NAME=  Its name on the panel [the city's, or the nearest city's]: "
if defined NAME set "NAME=%NAME:"=%"
if defined NAME set "NAME=%NAME:&=and%"
if defined NAME set "NAME=%NAME:|=%"
if defined NAME set "NAME=%NAME:<=%"
if defined NAME set "NAME=%NAME:>=%"
if defined NAME set "NAME=%NAME:^=%"
if not "%NAME%"=="" set "ARGS=%ARGS% --name "%NAME%""
rem the clock: the place's own time zone (summer time included), unless hours are given
set "TZIN="
set /p "TZIN=  Hours from UTC for the clock [the place's own clock, summer time included]: "
if defined TZIN set "TZIN=%TZIN:"=%"
if not "%TZIN%"=="" set "ARGS=%ARGS% --tz %TZIN%"

:ask_time
set "WHEN="
set /p "WHEN=  Start time in UTC, e.g. 2026-09-10T06:00 [now]: "
if defined WHEN set "WHEN=%WHEN:"=%"
if not "%WHEN%"=="" set "ARGS=%ARGS% --time %WHEN%"
set "OFF="
set /p "OFF=  Offline, no network (y/n) [n]: "
if defined OFF set "OFF=%OFF:"=%"
if /i "%OFF%"=="y" set "ARGS=%ARGS% --offline"
echo   Cloud model: auto none weather supercell squall bomex stratocumulus
echo                altocumulus altocumulus_ra altocumulus_un streets asperitas virga
set "SIM="
set /p "SIM=  Which [auto = what the weather calls for]: "
if defined SIM set "SIM=%SIM:"=%"
if not "%SIM%"=="" set "ARGS=%ARGS% --sim %SIM%"
set "GRID="
set /p "GRID=  Model grid: fast standard fine [standard with a GPU, fast without]: "
if defined GRID set "GRID=%GRID:"=%"
if not "%GRID%"=="" set "ARGS=%ARGS% --sim-size %GRID%"
set "QUAL="
set /p "QUAL=  Quality: low medium high photo [medium]: "
if defined QUAL set "QUAL=%QUAL:"=%"
if not "%QUAL%"=="" set "ARGS=%ARGS% --quality %QUAL%"
set "SIZE="
set /p "SIZE=  Window size [1280x760]: "
if defined SIZE set "SIZE=%SIZE:"=%"
if not "%SIZE%"=="" set "ARGS=%ARGS% --size %SIZE%"
call :run
goto menu

:selftest
echo.
echo   %PY% cloud_sim.py --selftest
%PY% cloud_sim.py --selftest
echo.
pause
goto menu

:run
echo.
echo   %PY% cloud_sim.py %ARGS%
echo.
%PY% cloud_sim.py %ARGS%
if errorlevel 1 (
    echo.
    echo   CloudSim stopped with an error - the message is above.
    pause
)
exit /b 0

:end
endlocal
