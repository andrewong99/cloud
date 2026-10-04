@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
rem Copyright (C) 2026 The CloudSim authors
rem CloudSim is free software: you can redistribute it and/or modify it under the
rem terms of the GNU General Public License, version 3 or (at your option) any
rem later version.  It comes with NO WARRANTY; see the LICENSE file.
rem Run CloudSim's 6471 checks.  Their windows are hidden; the loading
rem window shows for a moment while its own rule runs.
setlocal
cd /d "%~dp0"
python cloud_sim.py --selftest %*
echo.
pause
endlocal
