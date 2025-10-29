@echo off
echo 激活虚拟环境...
call venv\Scripts\activate

echo 启动可视化大屏应用...
python dashboard.app.py

pause