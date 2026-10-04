@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Copyright (C) 2026 The CloudSim authors
rem CloudSim is free software: you can redistribute it and/or modify it under the
rem terms of the GNU General Public License, version 3 or (at your option) any
rem later version.  It comes with NO WARRANTY; see the LICENSE file.
rem CloudSim launcher.  Keep this next to cloud_sim.py.
setlocal
cd /d "%~dp0"

python -c "import moderngl, pyglet, numpy" >nul 2>&1
if errorlevel 1 (
    echo CloudSim needs a few Python packages the first time you run it.
    echo Installing: moderngl pyglet numpy scipy pillow
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo The install failed.  Check that "python" is on your PATH.
        pause
        exit /b 1
    )
)

python cloud_sim.py %*
if errorlevel 1 pause
endlocal
