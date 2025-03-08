@echo off
echo 启动文件到视频转换系统...
echo.

:: 检查Python是否已安装
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python未安装，请先安装Python (3.8+)
    pause
    exit /b 1
)

:: 检查pip是否可用
python -m pip --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python pip模块未安装，请检查Python安装
    pause
    exit /b 1
)

:: 检查ffmpeg是否已安装
where ffmpeg > nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 未检测到FFmpeg，请确保FFmpeg已安装并添加到PATH环境变量
    echo 系统将尝试继续运行，但可能会失败...
    echo.
)

:: 检查是否需要安装依赖
if not exist venv (
    echo 首次运行，正在创建虚拟环境并安装依赖...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败，请检查网络连接并重试
        pause
        exit /b 1
    )
) else (
    call venv\Scripts\activate.bat
)

:: 启动主程序
echo 启动Web服务器，请在浏览器中访问以下地址:
echo.
echo     http://127.0.0.1:8080
echo.
echo 按Ctrl+C可停止服务器
echo.
python main.py

:: 如果程序异常退出
if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序异常退出，错误代码: %errorlevel%
    echo 请查看上方错误信息，或检查logs文件夹中的日志
    pause
)
