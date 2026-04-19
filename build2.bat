@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d F:\work\projects\relations\sdhg-gravity
cl /nologo /O2 /Fe:cdt_sim.exe cdt_sim.c
if exist cdt_sim.exe (echo BUILD OK) else (echo BUILD FAILED)
