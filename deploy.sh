#!/usr/bin/env bash
# 部署脚本 — 在 Linux 服务器上执行
# 用法：
#   bash deploy.sh                  # 自动检测 GPU，有则启用，不重建镜像
#   bash deploy.sh cpu              # 强制 CPU 模式，不重建镜像
#   bash deploy.sh gpu              # 强制 GPU 模式，不重建镜像
#   bash deploy.sh --build          # 自动检测 GPU，并重建镜像
#   bash deploy.sh gpu --build      # 强制 GPU 模式，并重建镜像
set -euo pipefail

MODE="auto"
BUILD=false

for arg in "$@"; do
    case "$arg" in
        auto|cpu|gpu)
            MODE="$arg"
            ;;
        --build)
            BUILD=true
            ;;
        *)
            echo "错误: 不支持的参数: $arg"
            echo "用法: bash deploy.sh [auto|cpu|gpu] [--build]"
            exit 1
            ;;
    esac
done
GPU_BASE_IMAGE="nvidia/cuda:11.8.0-runtime-ubuntu22.04"
GPU_PADDLE_VERSION="3.1.1"
GPU_PADDLE_CUDA_TAG="cu118"

set_env() {
    local key="$1"
    local value="$2"

    if grep -q "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        printf '\n%s=%s\n' "$key" "$value" >> .env
    fi
}

echo "=== 医疗单据脱敏服务 部署 ==="

# ── 1. 检查 Docker ──
if ! command -v docker &>/dev/null; then
    echo "[1/5] 安装 Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
else
    echo "[1/5] Docker 已安装 ✓"
fi

# ── 2. 检查 docker compose ──
if ! docker compose version &>/dev/null; then
    echo "错误: 需要 docker compose v2，请升级 Docker"
    exit 1
fi
echo "[2/5] Docker Compose 就绪 ✓"

# ── 3. 检查 .env ──
if [ ! -f .env ]; then
    echo "未找到 .env，从模板创建..."
    cp .env.example .env
    echo "请编辑 .env 填写 API Key 等配置后重新运行此脚本"
    echo "  vim .env"
    exit 1
fi
echo "[3/5] 配置文件就绪 ✓"

# ── 4. GPU 检测与配置 ──
HAS_GPU=false
if [ "$MODE" = "gpu" ]; then
    HAS_GPU=true
elif [ "$MODE" = "cpu" ]; then
    HAS_GPU=false
else
    # 自动检测
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        HAS_GPU=true
    fi
fi

if [ "$HAS_GPU" = true ]; then
    echo "[4/5] GPU 模式 — 检测到 NVIDIA GPU"

    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
        echo "错误: 已选择 GPU 模式，但主机无法正常执行 nvidia-smi"
        exit 1
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | xargs)
    echo "  GPU: ${GPU_NAME:-unknown} (${GPU_MEM_MB:-unknown} MiB)"

    # 检查 nvidia-container-toolkit
    if ! docker run --rm --gpus all "$GPU_BASE_IMAGE" nvidia-smi &>/dev/null 2>&1; then
        echo "  安装 NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
        distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
        curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
        apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
        if ! docker run --rm --gpus all "$GPU_BASE_IMAGE" nvidia-smi &>/dev/null 2>&1; then
            echo "错误: NVIDIA Container Toolkit 已安装，但 CUDA 11.8 容器仍无法访问 GPU"
            echo "请检查主机驱动与容器运行时是否兼容后重试"
            exit 1
        fi
        echo "  NVIDIA Container Toolkit 安装完成 ✓"
    else
        echo "  NVIDIA Container Toolkit 就绪 ✓"
    fi

    # 写入 .env GPU 参数（P4/8GB + 4核/20GiB 默认保守配置）
    set_env INSTALL_GPU true
    set_env BASE_IMAGE "$GPU_BASE_IMAGE"
    set_env PADDLE_GPU_VERSION "$GPU_PADDLE_VERSION"
    set_env PADDLE_CUDA_TAG "$GPU_PADDLE_CUDA_TAG"
    set_env OCR_POOL_SIZE 1
    set_env OCR_MODEL mobile
    set_env MAX_CONCURRENCY 2
    set_env DOCKER_CPUS 4
    set_env DOCKER_MEM 18G
    set_env DOCKER_SHM_SIZE 2gb
    set_env APP_WORKERS 1
    set_env OMP_NUM_THREADS 2
    set_env OPENBLAS_NUM_THREADS 1
    set_env MKL_NUM_THREADS 1
    set_env FLAGS_num_threads 1
    set_env FLAGS_fraction_of_gpu_memory_to_use 0.6

    echo "  已应用 P4 8GB 默认配置：OCR_POOL_SIZE=1, OCR_MODEL=mobile, APP_WORKERS=1, MAX_CONCURRENCY=2"
    echo "  机器规格匹配值：DOCKER_CPUS=4, DOCKER_MEM=18G"
    echo "  如需提速，请在稳定运行后逐步调高并发"

    COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.gpu.yml"
else
    echo "[4/5] CPU 模式"
    set_env INSTALL_GPU false
    set_env BASE_IMAGE python:3.11-slim

    COMPOSE_CMD="docker compose"
fi

# ── 5. 启动服务 ──
if [ "$BUILD" = true ]; then
    echo "[5/5] 重建镜像并启动..."
    $COMPOSE_CMD up -d --build
else
    echo "[5/5] 直接启动容器（不重建镜像）..."
    $COMPOSE_CMD up -d
fi

echo ""
echo "========================================="
echo "  部署完成！"
echo "========================================="
echo "  模式: $([ "$HAS_GPU" = true ] && echo "GPU" || echo "CPU")"
echo "  构建: $([ "$BUILD" = true ] && echo "已重建镜像" || echo "未重建镜像")"
echo "  地址: http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "  常用命令："
echo "    curl http://localhost:8000/health    # 健康检查"
if [ "$HAS_GPU" = true ]; then
echo "    $COMPOSE_CMD logs -f    # 查看日志"
echo "    $COMPOSE_CMD restart    # 重启服务"
echo "    $COMPOSE_CMD down       # 停止服务"
else
echo "    docker compose logs -f               # 查看日志"
echo "    docker compose restart               # 重启服务"
echo "    docker compose down                  # 停止服务"
fi
echo ""
echo "  调整资源：修改 .env 后执行"
echo "    docker compose up -d                 # 无需 rebuild"
echo "========================================="
