# 医疗单据隐私脱敏平台

基于 **OCR + Vision LLM** 的医疗单据敏感信息自动识别与打码服务。
当前以医疗票据、检查单、报告单等场景为主，支持从腾讯云 COS 读取图片，保留原图地址不变，并输出新的 `masked_url`，同时提供"仅检测"模式。

## 项目优势

- 四点定位打码：基于 OCR 文本行四点框生成遮罩区域，能跟随文字倾斜，而不是简单粗暴地盖一个大矩形。
- 方向自动纠正：内置 EXIF 方向识别、投影分析和 OCR 旋转重试，能处理扫描件、手机拍照造成的旋转、倒置和部分倾斜问题。
- 复杂拍摄场景更稳：针对弯曲、反光、压缩噪声、摩尔纹等真实拍摄干扰，采用 OCR 与 Vision LLM 结合的方式做交叉判断，降低漏检和误打码。
- 面向生产接入：保留原始 COS 链路，输出新文件地址，方便接入已有业务流程、审计链路和批处理任务。

> 由于项目面向医疗隐私场景，仓库不附带真实单据图片展示；README 只保留文字说明、接口示例和脱敏后的配置模板。

## 开源发布说明

- 仓库只保留脱敏后的示例配置，真实密钥必须放在本地 `.env` 或部署环境变量中。
- 不要把真实医疗图片、OCR 输出、日志或 `.env` 提交到版本库。
- 如果某个凭据曾经进入过 git 历史，发布前请先轮换该凭据，并重写历史后再公开仓库。
- 漏洞披露、敏感信息处理约定见 [SECURITY.md](./SECURITY.md)。

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

需要填写的关键配置：

| 变量 | 说明 |
|------|------|
| `DEFAULT_LLM_PROVIDER` | 默认提供商（`openai` / `claude` / `zhipu` / `bailian` / `volcengine` / `volcengine-lite`） |
| `COS_SECRET_ID` / `COS_SECRET_KEY` | 腾讯云 COS 密钥 |
| `COS_REGION` / `COS_BUCKET` | COS 区域与桶名 |
| `COS_DOMAIN_MAP` | 可选，自定义域名到 `bucket/region` 的 JSON 映射 |

> 如需兼容旧的同名覆盖链路，可在 `.env` 中启用：`CDN_REFRESH_ENABLED=true`。服务会在上传成功后尝试刷新 `masked_url` 对应的腾讯云 CDN 缓存；默认 best-effort，刷新失败不会影响主流程返回。

支持的 LLM 提供商（按需配置对应密钥）：
| 提供商 | 关键变量 | 示例模型 |
|--------|---------|---------|
| OpenAI 兼容 | `OPENAI_API_KEY` | `gpt-4o` |
| Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| 智谱 GLM | `ZHIPU_API_KEY` | `GLM-4.6V-Flash` |
| 阿里云百炼 | `BAILIAN_API_KEY` | `qwen3-vl-32b-instruct` |
| 火山引擎豆包 | `ARK_API_KEY` | `doubao-seed-1-8-251228` |

`llm_provider` 请求参数可以省略；省略时服务会使用 `.env` 中的 `DEFAULT_LLM_PROVIDER`。
如果你只是想先把整条链路跑通，`ZHIPU_MODEL=GLM-4.6V-Flash` 是一个合适的起点；按当前智谱官方文档，它标注为免费模型，适合初步联调和 smoke test。

### 3. 启动服务

```bash
python run.py
```

服务默认监听 `http://0.0.0.0:8000`，支持热重载。

### 4. 验证

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## 裸机部署（适合临时按量机器）

不需要 Docker，直接在服务器上安装运行，适合按量付费临时跑批，用完释放。

### 支持的系统

- Ubuntu / Debian（apt）
- TencentOS Server 3.x / CentOS / RHEL（dnf / yum）

### Python 要求

- **必须使用 Python 3.9 / 3.10 / 3.11**
- **不建议使用 Python 3.12**（Paddle/PaddleOCR 在服务器上容易出现依赖或运行时兼容问题）

### 一键部署

```bash
# 1. 上传到服务器
scp -r medical_tool/ user@your-server:/opt/medical_tool

# 2. 配置
cd /opt/medical_tool
cp .env.example .env
vim .env   # 填写 API Key、COS 配置

# 3. 一键部署（自动检测 GPU、启动服务）
bash deploy-bare.sh
```

也可以手动指定模式：

```bash
bash deploy-bare.sh gpu            # 强制 GPU 模式，不重装依赖
bash deploy-bare.sh cpu            # 强制 CPU 模式，不重装依赖
bash deploy-bare.sh gpu --build    # 强制 GPU 模式，并重装依赖
bash deploy-bare.sh cpu --build    # 强制 CPU 模式，并重装依赖
bash deploy-bare.sh stop           # 停止服务
```

### 脚本会自动完成

- 检测系统包管理器（apt / dnf / yum）
- 安装 Python 3.11（如果系统没有合适版本）
- 自动过滤掉不兼容的 Python 3.12
- 创建/重建虚拟环境 `.venv`
- 设置 `HF_ENDPOINT=https://hf-mirror.com` 以加速 Paddle 模型下载
- 在需要时重装 Python 依赖，并按 GPU 模式安装对应 `paddlepaddle-gpu`
- 后台启动 uvicorn，并以 `/health` 检查通过作为部署成功标准

### 常用运维命令

```bash
curl http://localhost:8000/health               # 健康检查
tail -f server.log                              # 查看日志
bash deploy-bare.sh stop                        # 停止服务
rm -rf .venv && bash deploy-bare.sh gpu --build # 虚拟环境损坏时重建
```

### TencentOS 3.x 注意事项

如果系统是 TencentOS Server 3.x，脚本会优先使用 `dnf` 并尝试安装 `python3.11`。这是推荐方式。

如果你之前已经用 Python 3.12 建过环境，务必先删掉旧环境：

```bash
rm -rf .venv
bash deploy-bare.sh gpu --build
```

---

## Docker 部署（推荐长期运行）

一套 Dockerfile + docker-compose.yml，自动检测 GPU，所有资源参数通过 `.env` 动态控制。

### 一键部署（3 步）

```bash
# 1. 上传到服务器
scp -r medical_tool/ user@your-server:/opt/medical_tool

# 2. 配置
cd /opt/medical_tool
cp .env.example .env
vim .env   # 填写 API Key、COS 配置

# 3. 部署（自动检测 GPU，有则启用）
bash deploy.sh
```

部署脚本会自动完成：检查/安装 Docker → 检测 GPU → 安装 NVIDIA Container Toolkit（如需）→ 配置 `.env` → 启动服务。

也可以手动指定模式：

```bash
bash deploy.sh gpu            # 强制 GPU 模式，不重建镜像
bash deploy.sh cpu            # 强制 CPU 模式，不重建镜像
bash deploy.sh gpu --build    # 强制 GPU 模式，并重建镜像
bash deploy.sh cpu --build    # 强制 CPU 模式，并重建镜像
```

### Docker GPU 服务器完整部署命令

如果你已经确认服务器有 NVIDIA GPU，推荐直接按下面顺序执行：

```bash
cd /opt/medical_tool
cp .env.example .env
vim .env
```

`.env` 中至少确认这些值：

```bash
INSTALL_GPU=true
BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04
PADDLE_GPU_VERSION=3.1.1
PADDLE_CUDA_TAG=cu118
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENDPOINT=https://hf-mirror.com

DOCKER_CPUS=4
DOCKER_MEM=18G
DOCKER_SHM_SIZE=2gb
APP_WORKERS=1
MAX_CONCURRENCY=2
OCR_POOL_SIZE=1
OCR_MODEL=mobile
FLAGS_fraction_of_gpu_memory_to_use=0.6
```

先验证宿主机 GPU：

```bash
nvidia-smi
```

再验证 Docker runtime 是否能访问 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

然后用 GPU compose 文件启动：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

启动后验证容器里的 GPU 与 Paddle：

```bash
docker exec -it medical-tool nvidia-smi
docker exec -it medical-tool python -c "import paddle; print('cuda=', paddle.device.is_compiled_with_cuda()); print('device=', paddle.device.get_device())"
```

查看日志：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
```

如果部署成功，日志里应看到类似：

```bash
paddle_cuda=True
paddle_device=gpu:0
OCR backend ready: ... paddle_device=gpu:0
```

> 注意：GPU 版后续的启动、重启、查看日志都必须继续带 `-f docker-compose.gpu.yml`。如果只执行普通 `docker compose up -d`，容易重新跑回 CPU 运行时。

验证：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## 服务器选购指南（80 并发）

### GPU 方案（推荐，跑批快）

| 项目 | 配置 |
|------|------|
| 机型 | 8 核 32G + T4 16G |
| 参考 | 腾讯云 GN7.2XLARGE32 / 阿里云 ecs.gn6i-c8g1.2xlarge |
| 按量价格 | ~10-15 元/时 |
| 系统盘 | 50G SSD |

部署：`bash deploy.sh gpu`（脚本自动配置 GPU 参数、安装 NVIDIA Container Toolkit）

`.env` 中手动设置：
```bash
DOCKER_CPUS=8
DOCKER_MEM=24G
MAX_CONCURRENCY=80
OCR_POOL_SIZE=4
OCR_MODEL=server
# 以下由 deploy.sh gpu 自动配置，无需手动改
# INSTALL_GPU=true
# BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04
```

### CPU 方案（无 GPU）

| 项目 | 配置 |
|------|------|
| 机型 | 16 核 64G |
| 参考 | 腾讯云 S5.4XLARGE64 / 阿里云 ecs.g7.4xlarge |
| 按量价格 | ~5-8 元/时 |
| 系统盘 | 50G SSD |

部署：`bash deploy.sh cpu`

`.env` 中手动设置：
```bash
DOCKER_CPUS=16
DOCKER_MEM=48G
MAX_CONCURRENCY=80
OCR_POOL_SIZE=16
OCR_MODEL=server
```

### 性能对比

| | GPU 方案 | CPU 方案 |
|--|---------|---------|
| 单张 OCR 耗时 | ~2-3s | ~10-15s |
| 单张总耗时（含 LLM） | ~30-40s | ~40-60s |
| 80 并发吞吐 | ~120-150 张/分钟 | ~80-100 张/分钟 |
| 按量费用 | ~12 元/时 | ~6 元/时 |

> 瓶颈在 LLM API 响应（30-60s），GPU 主要加速 OCR 阶段。跑批量大（万张+）时 GPU 总费用反而更低。

---

## 资源动态调整

所有参数在 `.env` 中，改完重启即可，**无需重新 build**：

```bash
vim .env                  # 修改参数
docker compose up -d      # 重启生效
```

### 场景配置速查

**跑批模式（高并发）：**
```bash
DOCKER_CPUS=16
DOCKER_MEM=32G
MAX_CONCURRENCY=80
OCR_POOL_SIZE=16
```

**日常模式（省资源/省钱）：**
```bash
DOCKER_CPUS=4
DOCKER_MEM=8G
MAX_CONCURRENCY=10
OCR_POOL_SIZE=4
```

**GPU 切换**（需 rebuild）：
```bash
bash deploy.sh gpu    # 自动配置 BASE_IMAGE、INSTALL_GPU 并 rebuild
```

### 常用运维命令

```bash
# CPU 模式
docker compose logs -f              # 查看实时日志
docker compose restart              # 重启服务
docker compose down                 # 停止服务
docker compose up -d                # 改 .env 后重启（资源调整）

# GPU 模式（需要加载 gpu 覆盖文件）
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart
docker compose -f docker-compose.yml -f docker-compose.gpu.yml stop
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

> 使用 `deploy.sh` 部署时无需记忆这些命令，脚本会自动选择正确的 compose 文件。
> 注意：GPU 模式不要直接执行 `docker compose up -d` 或 `docker compose restart`，否则不会加载 `docker-compose.gpu.yml`，可能重新回到 CPU 运行时。

---

## API 接口

### 健康检查

```
GET /health
```

返回：`{"status": "ok"}`

---

### 打码接口

```
POST /api/mask
Content-Type: application/json
```

**请求参数：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_url` | string | 是 | — | COS 上的图片 URL |
| `llm_provider` | string | 否 | `null` | LLM 提供商；为空时使用 `DEFAULT_LLM_PROVIDER`，支持 `openai` / `claude` / `zhipu` / `bailian` / `volcengine` / `volcengine-lite` |
| `mask_mode` | string | 否 | `"white"` | 打码方式：`white` / `black` / `gray` / `blur` / `mosaic` |
| `categories` | string[] | 否 | 全部 8 类 | 指定检测的敏感信息类别 |

**请求示例：**

```json
{
  "image_url": "https://your-bucket.cos.ap-nanjing.myqcloud.com/reports/medical_report.jpg",
  "llm_provider": "volcengine",
  "mask_mode": "white",
  "categories": ["name", "id_card", "phone"]
}
```

**成功响应：**

```json
{
  "success": true,
  "origin_url": "https://your-bucket.cos.ap-nanjing.myqcloud.com/reports/medical_report.jpg",
  "masked_url": "https://your-bucket.cos.ap-nanjing.myqcloud.com/reports/medical_report_masked_1710000000.jpg",
  "error": null
}
```

**处理流程：** 下载原图 → OCR + LLM 检测 → 打码 → 上传到新的 masked URL → 返回 `origin_url` + `masked_url`

---

### 检测接口

```
POST /api/detect
Content-Type: application/json
```

仅检测敏感信息，不执行打码。

**请求参数：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_url` | string | 是 | — | COS 上的图片 URL |
| `llm_provider` | string | 否 | `null` | LLM 提供商；为空时使用 `DEFAULT_LLM_PROVIDER`，支持 `openai` / `claude` / `zhipu` / `bailian` / `volcengine` / `volcengine-lite` |
| `categories` | string[] | 否 | 全部 8 类 | 指定检测的敏感信息类别 |

**请求示例：**

```json
{
  "image_url": "https://your-bucket.cos.ap-nanjing.myqcloud.com/reports/medical_report.jpg",
  "llm_provider": "volcengine",
  "categories": ["name", "phone"]
}
```

**响应：**

```json
{
  "sensitive_items": [
    {
      "text": "张三",
      "category": "name",
      "bbox": [100, 50, 200, 80],
      "ocr_line_index": 3
    }
  ],
  "total_lines": 25,
  "sensitive_count": 1
}
```

---

## 敏感信息类别

| 类别 | 标识 | 说明 |
|------|------|------|
| 患者姓名 | `name` | 2-4 个汉字的真实姓名 |
| 身份证号 | `id_card` | 15 或 18 位身份证号码 |
| 手机号码 | `phone` | 手机或固定电话号码 |
| 病历编号 | `medical_id` | 病历号/就诊号/住院号/超声号等 |
| 年龄 | `age` | 年龄数值（如 "33岁"） |
| 出生日期 | `date_of_birth` | 出生日期 |
| 医院名称 | `hospital` | 如 "XX市人民医院" |
| 医生姓名 | `doctor` | 报告医生、主治医生、审核医生等 |

## 打码方式

| 模式 | 标识 | 效果 |
|------|------|------|
| 白色覆盖 | `white` | 白色色块填充敏感区域（默认） |
| 黑色覆盖 | `black` | 黑色色块填充敏感区域 |
| 灰色覆盖 | `gray` | 灰色色块填充敏感区域 |
| 高斯模糊 | `blur` | 对敏感区域进行高斯模糊 |
| 马赛克 | `mosaic` | 对敏感区域进行像素化处理 |

---

## 项目结构

```
medical_tool/
├── run.py                          # 启动入口（uvicorn）
├── requirements.txt                # Python 依赖清单
├── Dockerfile                      # 统一镜像（通过 INSTALL_GPU 参数支持 CPU/GPU）
├── docker-compose.yml              # 编排配置（资源参数从 .env 读取）
├── deploy.sh                       # 一键部署脚本
├── .env.example                    # 环境变量模板
├── app/
│   ├── main.py                     # FastAPI 应用初始化、lifespan、健康检查
│   ├── config.py                   # Pydantic Settings 配置（读取 .env）
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py         # 组装 v1 版本路由
│   │       ├── deps.py             # 共享依赖注入（masker 工厂、并发控制）
│   │       └── endpoints/
│   │           ├── mask.py         # POST /mask — 打码接口
│   │           └── detect.py       # POST /detect — 检测接口
│   ├── core/
│   │   ├── cos.py                  # 腾讯云 COS 客户端封装（上传/下载/URL 解析）
│   │   ├── pipeline.py             # MedicalPrivacyMasker 主管道
│   │   ├── registry.py             # 插件注册表（OCR / 检测器 / 打码策略）
│   │   ├── models.py               # 内部数据模型（SensitiveRegion 等）
│   │   ├── ocr/                    # OCR 引擎（PaddleOCR，支持实例池化）
│   │   ├── detectors/              # 敏感信息检测器
│   │   │   ├── base.py             # 基类（规则提取 + LLM 检测 + CV 签名检测）
│   │   │   ├── openai.py           # OpenAI 兼容接口
│   │   │   ├── claude.py           # Anthropic Claude
│   │   │   ├── zhipu.py            # 智谱 GLM
│   │   │   ├── bailian.py          # 阿里云百炼
│   │   │   └── volcengine.py       # 火山引擎豆包
│   │   └── masking/                # 打码策略（blur / mosaic / solid）
│   ├── schemas/                    # 请求/响应 Pydantic 模型
│   ├── middleware/                  # 中间件（全局错误处理等）
│   └── utils/                      # 工具函数
└── tests/                          # 测试
```

---

## 扩展指南

### 新增 LLM 检测器

在 `app/core/detectors/` 下新建文件，继承 `BaseDetector` 并实现 `_call_vision()` 方法，用 `@detector_registry.register("name")` 注册，然后在 `__init__.py` 中导入即可。

### 新增打码策略

在 `app/core/masking/` 下新建文件，使用 `@masking_registry.register("name")` 装饰器注册即可。

### 新增 OCR 引擎

在 `app/core/ocr/` 下新建文件，使用 `@ocr_registry.register("name")` 装饰器注册即可。

### 新增功能模块（路由）

1. 在 `app/api/v1/endpoints/` 下新建文件
2. 在 `app/api/v1/__init__.py` 中注册路由

---

## 性能加速

### macOS — Apple Silicon (M1/M2/M3/M4)

PaddlePaddle 不支持 Apple Metal GPU 加速，但 ARM CPU 性能较强：

```bash
# 确保 ARM 原生 Python
python3 -c "import platform; print(platform.machine())"  # 应输出 arm64

# CPU 线程优化
export OMP_NUM_THREADS=8
```

### Linux — NVIDIA GPU

```bash
# 卸载 CPU 版，安装 GPU 版
pip uninstall paddlepaddle -y
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 验证
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"  # True
```

### Linux — 无 GPU（Intel CPU）

```bash
# 启用 MKL-DNN 加速（提速 2-3 倍）
export FLAGS_use_mkldnn=1
```
