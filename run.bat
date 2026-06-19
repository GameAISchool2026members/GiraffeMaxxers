@echo off
REM Launch the Game Jam app.
REM Prefers the `gamejam` conda env (if Anaconda is installed in the default
REM location); otherwise falls back to `python` from your active environment.
REM %USERPROFILE% keeps this portable — no hard-coded username.
set "CONDA=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if exist "%CONDA%" (
  "%CONDA%" run -n gamejam --no-capture-output python "%~dp0main.py"
) else (
  python "%~dp0main.py"
)
