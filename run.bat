@echo off
echo ========================================
echo    Starting AI Data Analyst...
echo ========================================
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
