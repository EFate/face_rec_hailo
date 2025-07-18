# Dockerfile

# --- Stage 1: 基础镜像 ---
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.10-slim

# --- Stage 2: 配置国内软件源 (清华源) ---
RUN rm -f /etc/apt/sources.list && \
    rm -rf /etc/apt/sources.list.d/* && \
    rm -rf /etc/apt/apt.conf.d/* && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    cat <<EOF > /etc/apt/apt.conf.d/99verify-peer.conf
Acquire::https::mirrors.tuna.tsinghua.edu.cn::Verify-Peer "false";
Acquire::https::mirrors.tuna.tsinghua.edu.cn::Verify-Host "false";
EOF

# 配置 pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip install --no-cache-dir --upgrade pip

# --- Stage 3: 安装系统依赖 ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Stage 4: 安装 Hailo 用户态运行时 ---
WORKDIR /app
COPY hailort_4.21.0_amd64.deb .

# [!!修正!!] 使用参考文件中的方法安装 .deb 包
# 1. 创建 .dockerenv 提示安装脚本在容器环境中运行
# 2. 使用 dpkg --unpack 先解压文件
# 3. 使用 dpkg --configure -a || true 来运行配置脚本并忽略任何错误
RUN touch /.dockerenv && \
    dpkg --unpack ./hailort_4.21.0_amd64.deb && \
    DEBIAN_FRONTEND=noninteractive dpkg --configure -a || true && \
    rm hailort_4.21.0_amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Stage 5: 安装项目依赖并拷贝代码 ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

# --- Stage 6: 配置容器运行 ---
EXPOSE 12010
EXPOSE 12011
CMD ["./start.sh"]