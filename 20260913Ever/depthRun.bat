@echo off
chcp 65001 > nul
title Depth Analyzer Launcher

:: 배치파일이 있는 현재 폴더로 이동
cd /d "%~dp0"

echo [ depth.py 실행 중... ]
echo.

:: 같은 폴더의 depth.py 실행
python depth.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [오류] depth.py 실행에 실패했습니다.
    echo 파이썬(Python)이 설치되어 있는지, 파일명이 depth.py가 맞는지 확인하세요.
    pause
)