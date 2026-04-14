#!/usr/bin/env bash
# 裸机部署脚本 — 不依赖 Docker，直接在服务器上安装并运行
# 适合按量付费临时机器，用完释放
#
# 用法：
#   bash deploy-bare.sh                  # 自动检测 GPU，不重装依赖
#   bash deploy-bare.sh cpu              # 强制 CPU 模式，不重装依赖
#   bash deploy-bare.sh gpu              # 强制 GPU 模式，不重装依赖
#   bash deploy-bare.sh --build          # 自动检测 GPU，并重装依赖
#   bash deploy-bare.sh gpu --build      # 强制 GPU 模式，并重装依赖
#   bash deploy-bare.sh stop             # 停止服务
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="auto"
BUILD=false
for arg in "$@"; do
    case "$arg" in
        auto|cpu|gpu|stop)
            MODE="$arg"
            ;;
        --build)
            BUILD=true
            ;;
        *)
            echo "错误: 不支持的参数: $arg"
            echo "用法: bash deploy-bare.sh [auto|cpu|gpu|stop] [--build]"
            exit 1
            ;;
    esac
done
VENV_DIR=".venv"
PID_FILE=".server.pid"
LOG_FILE="server.log"
PORT="${APP_PORT:-8000}"

if [ "$MODE" = "stop" ]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
            echo "服务已停止 (PID=$PID)"
        else
            echo "进程 $PID 已不存在"
        fi
        rm -f "$PID_FILE"
    else
        echo "未找到 PID 文件，尝试查找 uvicorn 进程组..."
        pkill -f "uvicorn app.main:app" 2>/dev/null || true
        pkill -f "multiprocessing.spawn" 2>/dev/null || true
        pkill -f "multiprocessing.resource_tracker" 2>/dev/null || true
        echo "已执行兜底清理"
    fi
    exit 0
fi

echo "=== 医疗单据脱敏服务 裸机部署 ==="

echo "[1/5] 检查系统依赖..."
if command -v dnf &>/dev/null; then
    PKG_MANAGER="dnf"
elif command -v yum &>/dev/null; then
    PKG_MANAGER="yum"
elif command -v apt-get &>/dev/null; then
    PKG_MANAGER="apt"
else
    echo "错误: 未找到支持的包管理器（dnf/yum/apt）"
    exit 1
fi
echo "  包管理器: $PKG_MANAGER"

install_pkg() {
    case "$PKG_MANAGER" in
        dnf) dnf install -y "$@" ;;
        yum) yum install -y "$@" ;;
        apt) apt-get update -qq && apt-get install -y -qq "$@" ;;
    esac
}

install_python311() {
    case "$PKG_MANAGER" in
        dnf)
            dnf install -y python3.11 python3.11-devel python3.11-pip 2>/dev/null || \
            dnf module install -y python39 2>/dev/null || \
            { dnf install -y epel-release && dnf install -y python3.11 python3.11-devel python3.11-pip; }
            ;;
        yum)
            yum install -y python3.11 python3.11-devel python3.11-pip 2>/dev/null || \
            { yum install -y epel-release && yum install -y python3.11 python3.11-devel python3.11-pip; }
            ;;
        apt)
            install_pkg software-properties-common
            add-apt-repository -y ppa:deadsnakes/ppa
            apt-get update -qq
            install_pkg python3.11 python3.11-venv python3.11-dev
            ;;
    esac
}

PYTHON_CMD=""
for cmd in python3.11 python3.10 python3.9; do
    if command -v "$cmd" &>/dev/null && "$cmd" -c "import sys; assert (3,9) <= sys.version_info[:2] < (3,12)" 2>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  安装 Python 3.11..."
    install_python311
    PYTHON_CMD="python3.11"
fi
echo "  Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"

if ! $PYTHON_CMD -m venv --help &>/dev/null; then
    $PYTHON_CMD -m ensurepip --upgrade 2>/dev/null || install_pkg python3-pip || true
fi

case "$PKG_MANAGER" in
    dnf|yum) install_pkg mesa-libGL libgomp gcc gcc-c++ make curl 2>/dev/null || true ;;
    apt)     install_pkg libgl1 libglib2.0-0 libgomp1 build-essential curl 2>/dev/null || true ;;
esac

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"
echo "  HF_ENDPOINT: $HF_ENDPOINT"
echo "  HF_HUB_ENDPOINT: $HF_HUB_ENDPOINT"
echo "[1/5] 系统依赖就绪 ✓"

echo "[2/5] 检查配置文件..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "已从模板创建 .env，请编辑后重新运行："
    echo "  vim .env"
    exit 1
fi
echo "[2/5] 配置文件就绪 ✓"

echo "[3/5] 配置 Python 环境..."
if [ -d "$VENV_DIR" ]; then
    if ! "$VENV_DIR/bin/python" -c "import sys; assert (3,9) <= sys.version_info[:2] < (3,12)" 2>/dev/null; then
        echo "  发现旧虚拟环境版本不兼容，删除重建..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "  虚拟环境已创建"
    BUILD=true
fi

source "$VENV_DIR/bin/activate"
if [ "$BUILD" = true ]; then
    python -m pip install --upgrade pip setuptools wheel
fi
echo "[3/5] Python 环境就绪 ✓"

echo "[4/5] 检查运行模式与依赖..."
HAS_GPU=false
if [ "$MODE" = "gpu" ]; then
    HAS_GPU=true
elif [ "$MODE" = "cpu" ]; then
    HAS_GPU=false
else
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        HAS_GPU=true
    fi
fi

if [ "$BUILD" = true ]; then
    echo "  重装 Python 依赖..."
    python -m pip install -r requirements.txt

    if [ "$HAS_GPU" = true ]; then
        echo "  检测到 GPU，安装 paddlepaddle-gpu（官方 CUDA 源，优先 cu118）..."
        python -m pip uninstall -y paddlepaddle 2>/dev/null || true
        python -m pip uninstall -y paddlepaddle-gpu 2>/dev/null || true

        CUDA_VER=""
        if command -v nvidia-smi &>/dev/null; then
            CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
        fi
        echo "  CUDA Version: ${CUDA_VER:-unknown}"

        GPU_INSTALLED=false
        for wheel_tag in cu118 cu126 cu128; do
            echo "  尝试安装 paddlepaddle-gpu==3.1.1 from ${wheel_tag} ..."
            if python -m pip install "paddlepaddle-gpu==3.1.1" -i "https://www.paddlepaddle.org.cn/packages/stable/${wheel_tag}/"; then
                GPU_INSTALLED=true
                break
            fi
        done

        if [ "$GPU_INSTALLED" != true ]; then
            echo "错误: paddlepaddle-gpu 安装失败，可能是驱动 / CUDA / wheel 不兼容"
            echo "可手动尝试："
            echo "  python -m pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/"
            exit 1
        fi
    else
        echo "  CPU 模式，保留 requirements.txt 中的 CPU 依赖"
    fi
else
    echo "  跳过依赖重装（如需重装请传 --build）"
fi

if [ "$HAS_GPU" = true ]; then
    if python -c "import paddle; assert paddle.device.is_compiled_with_cuda(); print(paddle.device.get_device())" 2>/dev/null; then
        echo "  GPU 验证通过 ✓"
    else
        echo "  警告: 当前环境未检测到可用的 Paddle GPU 运行时；如需重装请执行 bash deploy-bare.sh gpu --build"
    fi
    echo "  机器规格建议：P4 8GB / 4核 / 20GiB 保持 APP_WORKERS=1、OCR_POOL_SIZE=1、MAX_CONCURRENCY=2"
else
    echo "  CPU 模式"
fi
echo "[4/5] 环境检查完成 ✓"

echo "[5/5] 启动服务..."
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    kill "$OLD_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
fi

export FLAGS_enable_pir_api=0
export FLAGS_enable_pir_in_executor=0
export HF_ENDPOINT="$HF_ENDPOINT"
if [ "$HAS_GPU" = true ]; then
    export OCR_POOL_SIZE="${OCR_POOL_SIZE:-1}"
    export OCR_MODEL="${OCR_MODEL:-mobile}"
    export MAX_CONCURRENCY="${MAX_CONCURRENCY:-2}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export FLAGS_num_threads="${FLAGS_num_threads:-1}"
    export FLAGS_fraction_of_gpu_memory_to_use="${FLAGS_fraction_of_gpu_memory_to_use:-0.6}"
else
    export FLAGS_use_mkldnn=0
    export DNNL_MAX_CPU_ISA=AVX2
fi

# 生产启动统一走 app.main:app，由其负责应用级运行时初始化。
setsid "$VENV_DIR/bin/uvicorn" app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --limit-concurrency 500 \
    --timeout-keep-alive 120 \
    > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"

HEALTH_OK=false
for _ in $(seq 1 20); do
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        break
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        HEALTH_OK=true
        break
    fi
    sleep 1
done

if [ "$HEALTH_OK" = true ]; then
    echo ""
    echo "========================================="
    echo "  部署完成！"
    echo "========================================="
    echo "  模式: $([ "$HAS_GPU" = true ] && echo "GPU" || echo "CPU")"
    echo "  构建: $([ "$BUILD" = true ] && echo "已重装依赖" || echo "未重装依赖")"
    echo "  地址: http://$(hostname -I | awk '{print $1}'):${PORT}"
    echo "  PID:  $(cat "$PID_FILE")"
    echo "  日志: $SCRIPT_DIR/$LOG_FILE"
    echo ""
    echo "  常用命令："
    echo "    curl http://localhost:${PORT}/health    # 健康检查"
    echo "    tail -f $LOG_FILE                       # 查看日志"
    echo "    bash deploy-bare.sh stop                # 停止服务"
    echo "========================================="
else
    echo "启动失败！健康检查未通过，最近日志："
    tail -50 "$LOG_FILE"
    exit 1
fi