#!/bin/bash
# 位于项目根目录下的 start.sh

# 当任何命令失败时，立即退出脚本
set -e

# 关键步骤：切换到容器内的项目根目录 /app
# 这能确保后续所有命令的相对路径都是正确的
cd /app || exit

echo "[INFO] Current working directory: $(pwd)"
echo "[INFO] Starting Streamlit UI in background..."

# 以后台模式启动 Streamlit UI，并允许从外部访问
# 此路径 ui/ui.py 是相对于 /app 的，依然正确
streamlit run ui/ui.py --server.address=0.0.0.0 --server.port=12011 &

echo "[INFO] Starting FastAPI application in foreground..."

# 在前台启动 FastAPI 应用 (作为容器的主进程)
# 此路径 run.py 是相对于 /app 的，依然正确
python3 run.py --env production start