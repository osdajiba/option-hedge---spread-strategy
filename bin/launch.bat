@echo off
setlocal

REM 设置库文件夹路径
set "LIB_FOLDER=."

REM 添加conf、core和src文件夹到库文件夹路径
set "LIB_FOLDER=%LIB_FOLDER%;conf"
set "LIB_FOLDER=%LIB_FOLDER%;core"
set "LIB_FOLDER=%LIB_FOLDER%;src"

REM 运行main.py脚本
python -m bin.main
pause