@echo off
chcp 65001 >nul
REM ============================================
REM  개발 환경 자동 설정 스크립트
REM  프로젝트 루트(pyproject.toml 있는 위치)에서 실행하세요.
REM ============================================

echo [1/4] 가상환경 확인 중...
if exist ".venv" (
    echo   -^> .venv가 이미 존재합니다. 생성을 건너뜁니다.
) else (
    echo   -^> .venv 생성 중...
    python -m venv .venv
    if errorlevel 1 (
        echo [오류] venv 생성 실패. python이 PATH에 있는지 확인하세요.
        pause
        exit /b 1
    )
)

echo.
echo [2/4] 가상환경 활성화 및 pip 업그레이드...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/4] 메인 패키지 설치 중...
REM  PyPI의 Windows용 torch 휠은 CPU 전용이라 GPU가 있어도 가속되지 않는다.
REM  pip는 로컬 태그(+cpu / +cu130)를 무시해 버전만 같으면 재설치를 건너뛰므로,
REM  CPU 빌드가 이미 깔려 있으면 --force-reinstall 이 있어야 교체된다.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   -^> nvidia-smi가 없어 CPU용 torch로 진행합니다.
    goto skip_cuda
)

python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if not errorlevel 1 (
    echo   -^> CUDA용 torch가 이미 설치돼 있습니다.
    goto skip_cuda
)

echo   -^> GPU 감지됨. CUDA용 torch 설치 중...
pip install --force-reinstall --no-deps torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 echo   [경고] CUDA torch 설치 실패. CPU로 진행합니다.

:skip_cuda
pip install -e .
if errorlevel 1 (
    echo [오류] 메인 패키지 설치 실패. 위 에러 메시지를 확인하세요.
    pause
    exit /b 1
)

echo.
echo [4/4] 실행 패키지 설치 중...
if exist "runs\pyproject.toml" (
    pip install -e .\runs
    if errorlevel 1 (
        echo [오류] 실행 설치 실패. 위 에러 메시지를 확인하세요.
        pause
        exit /b 1
    )
) else (
    echo   -^> runs\pyproject.toml이 없어 건너뜁니다.
)

echo.
echo ============================================
echo  설정 완료!
echo  이후 터미널에서는 아래 명령으로 활성화하세요:
echo    .venv\Scripts\activate.bat
echo ============================================
pause