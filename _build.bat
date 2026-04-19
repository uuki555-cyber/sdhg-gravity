@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >/dev/null 2>&1
cd /d F:\work\projects\relations\sdhg-gravity
cl /nologo /O2 /Fe:cdt_sim.exe cdt_sim.c
