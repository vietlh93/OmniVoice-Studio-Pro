@echo off
:: Di chuyển dấu nhắc lệnh đến đúng thư mục chứa file .bat này
cd /d "%~dp0"

:: 1. Kiểm tra và kích hoạt venv
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo [ERROR] Khong tim thay thu muc venv tai: %~dp0venv
    pause
    exit /b
)

:: 2. Chạy ứng dụng Python (Gradio sẽ tự động mở trình duyệt khi khởi tạo thành công)
python app.py

pause