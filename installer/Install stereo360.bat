@echo off
rem  Double-click this. It does not matter which folder it is in -- the
rem  installer picks its own location and downloads everything it needs.
rem
rem  This wrapper exists because a .ps1 cannot be double-clicked: Windows
rem  opens it in Notepad, and a script downloaded from the internet carries
rem  a Mark-of-the-Web that the default execution policy refuses to run.
rem  Calling PowerShell directly with -ExecutionPolicy Bypass sidesteps both,
rem  and a .bat is something Explorer will actually launch.

setlocal
title Installing stereo360

if not exist "%~dp0install.ps1" (
    echo.
    echo   install.ps1 is missing. It must sit next to this file.
    echo   Re-download both from the releases page and keep them together.
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo   Finished. You can close this window.
) else (
    echo   The installer stopped with error %RC%.
    echo   The messages above say where. Nothing outside the install folder
    echo   was changed, so it is safe to fix the problem and run this again.
)
echo.
pause
exit /b %RC%
