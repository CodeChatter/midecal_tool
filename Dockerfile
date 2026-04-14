# 通过 BASE_IMAGE 参数切换 CPU/GPU 基础镜像
# CPU: python:3.11-slim（默认）
# GPU: nvidia/cuda:11.8.0-runtime-ubuntu22.04（推荐，匹配 Paddle cu118）
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG INSTALL_GPU=false
ARG PADDLE_GPU_VERSION=3.1.1
ARG PADDLE_CUDA_TAG=cu118

# 国内镜像加速（Debian trixie / bookworm 通用）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null \
    || sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null \
    || true

# GPU 镜像（基于 ubuntu）需要手动装 Python
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            software-properties-common \
        && add-apt-repository -y ppa:deadsnakes/ppa \
        && apt-get update && apt-get install -y --no-install-recommends \
            python3.11 python3.11-venv python3.11-dev python3-pip \
        && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
        && ln -sf /usr/bin/python3 /usr/bin/python \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# 系统依赖（PaddleOCR 需要 libGL + libglib，healthcheck 需要 curl）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 禁用 PaddlePaddle PIR 模式
ENV FLAGS_enable_pir_api=0
ENV FLAGS_enable_pir_in_executor=0

# 安装 Python 依赖
COPY requirements.txt .
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt \
        && pip uninstall -y paddlepaddle paddlepaddle-gpu \
        && pip install --no-cache-dir \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            --extra-index-url "https://www.paddlepaddle.org.cn/packages/stable/${PADDLE_CUDA_TAG}/" \
            "paddlepaddle-gpu==${PADDLE_GPU_VERSION}"; \
    else \
        pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt; \
    fi

# 复制源码
COPY . .

EXPOSE 8000

# 生产启动：默认单 worker，避免小显存 GPU 上多进程重复预热 OCR。
CMD ["sh", "-c", "gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w ${APP_WORKERS:-1} --bind 0.0.0.0:8000 --timeout 300 --graceful-timeout 120 --keep-alive 120"]
