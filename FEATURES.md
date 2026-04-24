# 医疗单据隐私脱敏平台 · 功能文档

> 一份"功能 + 处理流程 + 接口 + 配置 + 扩展点"合一的综合文档。
> 快速上手看 `README.md`，接口字段细节看 `API.md`，本文聚焦"这个服务到底能做什么、怎么做的"。

---

## 目录

- [1. 项目定位](#1-项目定位)
- [2. 能力总览](#2-能力总览)
- [3. 系统架构](#3-系统架构)
- [4. 完整处理流程](#4-完整处理流程)
- [5. 核心能力详解](#5-核心能力详解)
  - [5.1 OCR 引擎（PaddleOCR + 子进程池）](#51-ocr-引擎paddleocr--子进程池)
  - [5.2 图像方向自动矫正](#52-图像方向自动矫正)
  - [5.3 Vision LLM 多提供商](#53-vision-llm-多提供商)
  - [5.4 敏感信息分类体系](#54-敏感信息分类体系)
  - [5.5 规则 + CV 兜底检测](#55-规则--cv-兜底检测)
  - [5.6 人脸 / 人像打码](#56-人脸--人像打码)
  - [5.7 打码策略](#57-打码策略)
  - [5.8 非文字单据直出](#58-非文字单据直出)
  - [5.9 腾讯云 COS 适配](#59-腾讯云-cos-适配)
  - [5.10 并发控制与稳定性](#510-并发控制与稳定性)
- [6. API 接口一览](#6-api-接口一览)
- [7. 配置项全集](#7-配置项全集)
- [8. 扩展点](#8-扩展点)
- [9. 性能与运维](#9-性能与运维)
- [10. 已知边界与限制](#10-已知边界与限制)

---

## 1. 项目定位

面向**医疗单据**场景（但通用于任意图文单据）的**敏感信息自动识别 + 打码**服务。

- **输入**：腾讯云 COS 上的单据图片 URL
- **输出**：同桶新 key 的打码图 URL（原图不改动）
- **识别方式**：OCR 文本定位 + Vision LLM 语义判定 + 规则 / CV 兜底
- **部署形态**：FastAPI 单进程服务，支持裸机 / Docker / CPU / GPU

---

## 2. 能力总览

| 大类 | 能力 | 说明 |
|------|------|------|
| 核心接口 | `/api/mask` | 下载 → 检测 → 打码 → 回传新 URL，原图保留 |
| 核心接口 | `/api/detect` | 仅检测，返回敏感项清单（text / category / bbox / 所在 OCR 行） |
| 核心接口 | `/api/ocr` | 纯 OCR 文字提取，返回每行文本 + bbox + 置信度 |
| 运维 | `/health` | 健康检查 |
| 图像预处理 | EXIF + 投影方差方向矫正 | 手机拍照、扫描件自动摆正 |
| 图像预处理 | OCR 结果驱动的 90°/180°/270° 旋转重试 | 竖排文本 / 乱码自动回滚最优方向 |
| OCR | PaddleOCR 子进程池 | 子进程隔离 C++ SIGSEGV，自动重启，池化并发 |
| OCR | `server` / `mobile` 模型自由切换 | 精度 vs. 速度按场景可选 |
| Vision LLM | 5 家多模态模型可插拔 | OpenAI / Claude / 智谱 GLM / 阿里百炼 / 火山豆包（含 lite） |
| 敏感类别 | 10 类隐私字段 | 姓名 / 身份证 / 电话 / 病历号 / 医生 / 邮箱 / 地址 / 医院 / 年龄 / 日期 / 出生日期 / 人像 / 手写签名 |
| 规则兜底 | 医生 / 编号 / 采样人正则匹配 | LLM 漏判时从 OCR 文本直接补齐 |
| CV 兜底 | 底部手写签名 / 医生附近签名 / 同位置上一行签名 | OCR 识别不到的手写笔迹也能覆盖 |
| CV 能力 | 人脸检测（YuNet / Haar 自动回退） | portrait 类别专用 |
| 打码策略 | 白 / 黑 / 灰 / 高斯模糊 / 马赛克 | 多边形级遮罩，跟随文字倾斜 |
| 场景兼容 | 非文字单据原图直出 | CT / MRI / 超声等无文字图跳过脱敏 |
| 存储 | 腾讯云 COS | 支持标准域名 + 自定义域名映射（JSON 配置） |
| 部署 | 裸机 / Docker / GPU | `deploy-bare.sh` + `deploy.sh` + `docker-compose*.yml` |
| 稳定性 | 全局信号量 + 线程池 | `MAX_CONCURRENCY` 硬上限；OCR 子进程异常自恢复 |
| 可观测 | 每阶段耗时日志 + X-Request-ID | pipeline 各阶段 ms 级指标逐行记录 |

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          FastAPI (uvicorn)                                │
│  /health  /api/mask  /api/detect  /api/ocr                                │
└──────────────┬──────────────────────────────────────────────┬────────────┘
               │ RequestLoggingMiddleware + CORS + 错误处理    │
               ▼                                              ▼
     ┌────────────────────┐                        ┌────────────────────┐
     │  get_semaphore()   │ ── 全局并发控制 ─→     │  ThreadPoolExecutor │
     │  asyncio.Semaphore │                        │  (MAX_CONCURRENCY)  │
     └────────────────────┘                        └──────────┬─────────┘
                                                              │ 同步任务
                                                              ▼
                                         ┌────────────────────────────────┐
                                         │   MedicalPrivacyMasker         │
                                         │   (pipeline.py 主管道)          │
                                         └────────────────────────────────┘
                                               │         │          │
                         ┌─────────────────────┘         │          └──────────────────┐
                         ▼                               ▼                             ▼
             ┌────────────────────┐       ┌────────────────────────┐        ┌────────────────────┐
             │ OrientationCorrector│       │  BaseOCREngine         │        │  BaseDetector       │
             │ EXIF + 投影方差     │       │  PaddleOCR (子进程池)   │        │  Vision LLM +       │
             └────────────────────┘       └────────────────────────┘        │  规则 + CV 兜底     │
                                                                            └──────────┬─────────┘
                                                                                       │
                                                                  ┌────────────────────┼────────────────────┐
                                                                  ▼                    ▼                    ▼
                                                       ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
                                                       │ SolidColor       │  │ BlurStrategy     │  │ MosaicStrategy   │
                                                       │ (white/black/gray)│  │ 高斯模糊          │  │ 马赛克            │
                                                       └──────────────────┘  └──────────────────┘  └──────────────────┘
```

**插件化三件套**（`app/core/registry.py`）

| 注册表 | 注册内容 | 使用方 |
|--------|----------|--------|
| `ocr_registry` | OCR 引擎（当前内置 `paddle`） | `get_ocr_engine()` |
| `detector_registry` | Vision LLM 检测器 | `get_masker(provider)` 按 key 取 |
| `masking_registry` | 打码策略 | `get_masker(..., mode)` / `/api/mask` 参数 `mask_mode` |

---

## 4. 完整处理流程

以 `POST /api/mask` 为例：

```
┌── 请求进入 ──────────────────────────────────────────────────────────┐
│  1. 校验 mask_mode / 扩展名 / 解析 COS URL                           │
│  2. 全局信号量 + 线程池                                              │
└─────────────────────────────┬────────────────────────────────────────┘
                              ▼
┌── 同步任务 _mask_sync ───────────────────────────────────────────────┐
│  STEP 1/3  下载 COS 原图到临时目录 + 魔数校验                        │
│  STEP 2/3  MedicalPrivacyMasker.process_image                        │
│           ├ 阶段 0  方向矫正（EXIF → 投影方差 → cv2.rotate）         │
│           ├ 阶段 1  OCR                                              │
│           │       └ 空文本 → 非文字单据，原图直出                     │
│           ├ 阶段 1.2  有效文本判定（<2 行 或 全短文本 → 非文字单据） │
│           ├ 阶段 1.5  OCR 质量评估                                   │
│           │       ├ 竖向文本 → 旋转 90°/270°/180° 重试               │
│           │       └ 乱码/行少 → 旋转 90°/270°/180° 重试              │
│           ├ 阶段 1.6  再次有效性校验                                 │
│           ├ 阶段 2   Vision LLM + 规则 + CV 五路检测                 │
│           │       ├ LLM 检测 (图 + OCR JSON → 敏感项列表)             │
│           │       ├ 规则匹配 (医生/编号/采样人正则)                    │
│           │       ├ 底部手写签名 CV                                   │
│           │       ├ 医生附近签名 CV                                   │
│           │       ├ 同位置上一行签名 CV                               │
│           │       └ 人脸检测 (YuNet / Haar)                           │
│           │       → 去重（IoU > 0.3 且区域较大者保留）                │
│           └ 阶段 3  打码（多边形 mask_strategy.apply）                │
│  STEP 3/3  上传到 {key}_masked_{timestamp}{ext} → 返回新 URL         │
└──────────────────────────────────────────────────────────────────────┘
```

**`/api/detect` 流程**：相同的阶段 0 → 1 → 2，但不走阶段 3，返回敏感项 JSON。
**`/api/ocr` 流程**：只跑 OCR，返回行级结果。

---

## 5. 核心能力详解

### 5.1 OCR 引擎（PaddleOCR + 子进程池）

**实现文件**：`app/core/ocr/paddle.py`

**关键设计**：
- **子进程隔离**：PaddlePaddle C++ 层在部分 Docker + v4 模型下会触发 SIGSEGV，直接杀 worker 进程。本服务把 OCR 推理放到独立 `multiprocessing.Process`，崩溃只会杀子进程，主进程自动重启 worker。
- **Worker 池**：大小由 `OCR_POOL_SIZE` 控制（0 = 自动：GPU 下 1，CPU 下 `cpu_count - 1`）。通过 `queue.Queue` 做公平取用。
- **模型切换**：`OCR_MODEL` 可选 `server`（PP-OCRv5_server，精度高）或 `mobile`（PP-OCRv5_mobile，速度快）。
- **双 API 兼容**：优先 `predict()`（新版 PaddleX 管线，异常可捕获），失败兜底 `ocr(cls=True)` / `ocr()`。
- **缩放优化**：超过 960px 最长边的图自动按比例缩到 960，OCR 完成后反向映射 bbox 回原图坐标，在保持精度的前提下显著降低推理耗时。
- **预热机制**：`prewarm_ocr_engine()` 在 lifespan 阶段对每个 worker 跑一张 64×64 白图触发 C++ 运行时初始化，避免首个真实请求超时。用 `/tmp/medical_tool_ocr_prewarm.lock` 做跨进程互斥。
- **显存回收**：GPU 模式下每次 OCR 完 `paddle.device.cuda.empty_cache()`，防止长时间跑批显存碎片化。
- **超时 / 过载识别**：推理 120s 超时；错误字符串命中 `out of memory / cublas_status / cudnn_status` 等关键字时抛 `SystemOverloadError` → API 返回"当前系统压力过大，请稍后重试"。

### 5.2 图像方向自动矫正

**实现文件**：`app/core/pipeline.py` 中 `OrientationCorrector` 类 + pipeline 的 "阶段 1.5"。

**两层策略**：

1. **阶段 0 — 启动时矫正**（零成本优先）
   - **EXIF Orientation 标签**：手机拍照带元数据时直接读 tag 274 → 映射 `{3: 180, 6: 90, 8: 270}`。
   - **水平投影方差**：无 EXIF 时的兜底。把图缩到 800px 短边，做 OTSU 二值化，比较 0° vs. 180° 的行投影方差——正方向文字"行密→行稀"的方差更大。**不参与 90°/270° 判定**，因为宽幅横向图会被系统性误判。

2. **阶段 1.5 — OCR 结果驱动的旋转重试**
   - 跑完第一次 OCR 后，评估是否需要再转：
     - 长文本行中 `bbox` 高 > 宽 ×1.5 占比过半 → 判定为"竖向文本"，图片应该转 90° 或 270°。
     - 行数 < 4 或平均字符数 < 3 → 判定为"OCR 乱码"，图片可能上下颠倒。
   - 依次尝试 90° / 270° / 180°，每次重跑 OCR，按"竖向修正 > 行数显著增加 > 乱码变正常"规则选最优角度。
   - 最优若仍不比原图强 → 回滚到原方向，避免越改越糟。
   - 矫正完成后若 OCR 结果仍稀薄（< 2 行且无长文本），判定为"非文字单据"，走**原图直出**（见 5.8）。

### 5.3 Vision LLM 多提供商

**实现文件**：`app/core/detectors/` 目录。

| provider 值 | 对接模型（默认） | 关键配置 |
|-------------|------------------|----------|
| `openai` | `gpt-4o` 或任意 OpenAI 兼容模型 | `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_VISION_DETAIL` |
| `claude` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` |
| `zhipu` | `GLM-4.6V-Flash` | `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` / `ZHIPU_MODEL` |
| `bailian` | `qwen3-vl-32b-instruct` | `BAILIAN_API_KEY` / `BAILIAN_BASE_URL` / `BAILIAN_MODEL` |
| `volcengine` | `doubao-seed-1-8-251228` | `ARK_API_KEY` / `ARK_BASE_URL` / `ARK_MODEL` |
| `volcengine-lite` | `doubao-seed-2-0-lite-260215`（低时延 lite 版） | `ARK_API_KEY` / `ARK_MODEL_LITE` |

**统一特征**：
- HTTP 连接复用：`httpx.HTTPTransport` 连接池大小 = `MAX_CONCURRENCY + 20`，keepalive = `MAX_CONCURRENCY`。
- 超时：300s（读）+ 15s（连接），无 SDK 侧重试，依赖上层应用语义判定。
- 请求构造：`system` 写动态拼装的类别 prompt，`user` 带 `data:{mime};base64,...` 图 + OCR JSON 文本。
- 图像预处理：超过 `VISION_IMAGE_MAX_SIDE`（默认 1280）的图按比例缩放并重新 JPEG 压缩（`VISION_JPEG_QUALITY`，默认 75），减少 LLM 上行带宽。
- 响应 `max_tokens`：`VISION_MAX_TOKENS`（默认 384），压低响应延迟。
- 温度固定 0.1，确保结构稳定。

**Prompt 工程要点**（`app/core/detectors/base.py::build_system_prompt`）：
- 明确"只标注字段【值】，不标注标签"。
- 医务人员 vs. 患者姓名强制区分（`doctor` vs. `name`）。
- 手写签名用特殊 JSON 格式 `{"type": "handwritten_signature", "bbox_pct": [x1,y1,x2,y2]}` 返回，percentage 单位。
- `date` 类按需开启（不在请求 `categories` 时 prompt 明确排除日期）。
- 强制 JSON-only 输出，服务端仍做 markdown 代码块剥离兜底。

### 5.4 敏感信息分类体系

**13 种类别**（`categories` 参数不传时使用"默认 9 类"，即第一列 `默认` = ✅ 的行）：

| category 值 | 默认 | 说明 |
|-------------|------|------|
| `name` | ✅ | 患者 / 受检者 / 家属 / 联系人 / 采样人 / 送检人 / 录入人等 **非医务人员** 姓名 |
| `id_card` | ✅ | 15 / 18 位身份证号 |
| `phone` | ✅ | 手机 + 座机号码 |
| `medical_id` | ✅ | 病历号 / 就诊号 / 住院号 / 超声号 / 门诊号 / 检查号 / 条码号 / 仪器号等 |
| `doctor` | ✅ | 所有医务人员姓名（申请 / 报告 / 审核 / 主治 / 经治 / 检查 / 技师 / 操作者等） |
| `email` | ✅ | 邮箱地址 |
| `address` | ✅ | 家庭 / 户籍 / 通讯 / 居住地址 |
| `hospital` | ✅ | 医院 / 分院 / 院区 / 诊所名称（含单据顶部标题） |
| `portrait` | ✅ | 真人人像（证件照 / 头像 / 脸部特写 / 半身照），不含卡通 / 插画 / 医学影像 |
| `age` |  | 年龄数值（如 "33岁"） |
| `date` |  | 所有日期格式（检查日期 / 报告日期 / 打印日期 / 入出院日期 / 手术日期等） |
| `date_of_birth` |  | 出生日期 |
| `handwritten_signature` | — | 手写签名（不需要传入 categories，规则 + LLM + CV 自动识别） |

> 注：`handwritten_signature` 是**服务端内置的输出类别**，前端不需要在 `categories` 里传；无论请求什么，只要图里有手写笔迹都会被打码。

### 5.5 规则 + CV 兜底检测

即使 LLM 漏判或 OCR 漏识，服务端还有多路兜底（全部在 `BaseDetector.detect()` 内叠加）：

#### 规则匹配（`_extract_rule_based`）

三条正则，从 OCR 文本直接提取：

| 规则 | 匹配模式 | 产出 category |
|------|----------|---------------|
| `_DOCTOR_FIELD_RE` | `(报告\|审核\|主治\|经治\|检查\|主管\|…)?(医生\|医师\|技师\|操作者\|检查者)：XXX` | `doctor` |
| `_MEDICAL_ID_RE` | `(报告单号\|病历号\|就诊号\|住院号\|超声号\|…\|仪器编号)：XXX` | `medical_id` |
| `_STAFF_NAME_RE` | `(采样人\|送检人\|送检者\|核收人\|录入人)：XXX` | `name` |

规则匹配到的值会用 `get_value_polygon` 计算精确到字符的倾斜多边形（跟随文字方向），不会超打。

#### 空医生标签扩展（`_mask_empty_doctor_labels`）

当 OCR 行只有 "检查医生：" 这种标签却没有姓名值（手写被 OCR 漏识）时，向右侧扩展 0.7 倍行宽、上下 0.3 倍行高的区域做遮罩，但前提是**扩展区域里没有高置信度 OCR 文本**（避免覆盖到下一个字段的印刷内容）。

#### "手签："扩展（`_extract_handwritten_signatures`）

匹配 `手签：` 标签后，从冒号位置开始向右大幅扩展（整行宽），上下各 0.5 倍行高，覆盖手写签名区。`cv2.fillPoly` 自动裁剪到图片边界。

#### 底部手写签名 CV（`_detect_signature_cv`）

截取图像底部（最后一行 OCR 之下的留白区），做 OTSU 二值化 + 轮廓分析。**双层阈值策略**：

- **关键词模式**：OCR 中出现 `签名 / 手签 / 签字 / 签章 / 盖章` → 降低阈值（填充率 2%–45%，宽 ≥ 20px）。
- **纯视觉模式**：无关键词 → 严格阈值（笔画 ≥ 4 个；宽度 ≤ 底部 40%；填充率 3%–35%；不贴底边 5%；尺寸 ≥ 30×15）。

识别到则产出 `handwritten_signature` 遮罩。

#### 医生附近签名 CV（`_detect_signature_near_doctors`）

对每个已识别出的医生姓名，在其**正下方 2.5 倍行高** + **左右各 0.5 倍行高**的矩形内找手写笔迹。若区域内已有高置信度 OCR 文本则跳过（说明是印刷体）。

#### 同水平位置上一行签名 CV（`_detect_sibling_row_signature`）

专门针对"检查医师：XXX / 审核医师：YYY"上下两行双医师结构——若上一行被印章遮挡导致 OCR 失败，用下一行的 bbox 反推上一行的姓名槽位置，在该区域扫手写笔迹。**只在下半图触发**，避免误扫正文。

#### 手写签名 CV 二次校验（`_verify_handwriting_cv`）

LLM 返回的 `handwritten_signature` bbox 不可完全信任（可能给出的是印刷区），所以每个 LLM 报告的签名区域还会再跑一次轮廓 + 填充率校验（≥ 2 笔画且填充率 1%–45%），通不过直接丢弃。

#### 去重（`_deduplicate`）

所有路来的 `SensitiveRegion` 最后按 bbox 面积降序遍历，IoU > 0.3 的被吸收进大框，保证不重复打码。

### 5.6 人脸 / 人像打码

**实现文件**：`app/core/face_detector.py`

- **默认后端**：YuNet（ONNX，精度高），模型路径 `models/face_detection_yunet.onnx`（仓库内已附 233KB ONNX 文件）。
- **回退**：OpenCV `haarcascade_frontalface_default.xml`（精度较低，但随 OpenCV 安装自带，永远可用）。
- **推理**：纯 CPU，不占 GPU 显存。
- **过滤规则**：短边小于 `min_size_ratio × 图片短边`（默认 0.02）的小人脸被判为噪声忽略。
- **外扩 padding**：检测到人脸后向外扩宽 20% / 上下 30%，覆盖头发 / 帽子 / 颈部，保证隐私。
- **不依赖 LLM**：不管 LLM 是否报告 `portrait`，CV 人脸检测始终运行，作为独立一路参与最后 de-dup。

**自定义**：可通过环境变量 `FACE_YUNET_MODEL` 指定外部 ONNX 模型路径。

### 5.7 打码策略

**实现文件**：`app/core/masking/`

| `mask_mode` | 类 | 效果 | 实现要点 |
|-------------|---------------|------|----------|
| `white` | `SolidColorStrategy((255,255,255))` | 白色填充 | `cv2.fillPoly` 多边形填充 |
| `black` | `SolidColorStrategy((0,0,0))` | 黑色填充 | 同上 |
| `gray` | `SolidColorStrategy((128,128,128))` | 灰色填充 | 同上 |
| `blur` | `BlurStrategy` | 高斯模糊 | 核大小自适应 `max(35, min(h,w)//2)`，多边形 mask + ROI 原地替换 |
| `mosaic` | `MosaicStrategy` | 像素化马赛克 | 先降采样到 `w/15, h/15` 再最近邻放大回原尺寸 |

所有策略都作用在 **多边形** 上（而非外接矩形），遮罩会跟随倾斜文字的实际朝向，视觉上更贴合。

### 5.8 非文字单据直出

为了覆盖 CT / MRI / 超声 / 眼底照片等"图里没有文字"的场景，pipeline 在**两个检查点**判定"无需脱敏"直接原图拷贝输出：

1. OCR 结果为空 → 直出。
2. OCR 少于 2 行，或 2 行以上但全部短于 3 字符（噪点） → 判定非文字单据，直出。
3. 走完旋转重试后第二次检查同样判据 → 直出。

对应日志 `[Pipeline] 影像图直接输出（未做脱敏）`。`masked_url` 仍然会产出，只是内容和原图一致。

### 5.9 腾讯云 COS 适配

**实现文件**：`app/core/cos.py`

- **URL 解析**：
  - 标准格式 `https://<bucket>.cos.<region>.myqcloud.com/<key>`（同时支持 `tencentcos.cn`）。
  - **自定义域名映射**：在 `.env` 里配置 `COS_DOMAIN_MAP`（JSON），示例：
    ```json
    {"cos-assistant.unilifetech.com": "assistant-1319085945/ap-nanjing"}
    ```
    服务自动反查域名→(bucket, region) 进行下载与构造返回 URL。
- **按 region 缓存 client**：多 region 多桶在同进程内共享连接池（`PoolConnections=150, PoolMaxSize=150`）。
- **输出 key 约定**：`<stem>_masked_<时间戳>.<ext>`，原图 key 绝不被覆盖。
- **其他工具**：`build_backup_key`、`generate_presigned_url`（按需签出临时下载链接，默认 10 分钟）。

### 5.10 并发控制与稳定性

| 机制 | 位置 | 作用 |
|------|------|------|
| `asyncio.Semaphore(MAX_CONCURRENCY)` | `deps.get_semaphore()` | 全局同时处理的请求数硬上限 |
| `ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)` | `deps._get_thread_pool()` | 把阻塞型同步任务（下载 / OCR / 上传）卸载到线程池，不阻事件循环 |
| OCR 子进程池 | `PaddleOCREngine` | 进一步把 OCR 推理关到子进程里，隔离 SIGSEGV |
| `SystemOverloadError` | `app/core/errors.py` | OOM / cublas / 超时等 → 业务层返回"系统压力过大，请稍后重试"而非 500 |
| `RequestLoggingMiddleware` + `X-Request-ID` | `middleware/error_handler.py` | 每请求日志对齐 + 响应头返回 8 位 request id 便于追踪 |
| 全局异常处理 | 同上 | `ValueError → 400`；其他未处理异常 → 500 "内部服务器错误" |
| `prewarm_ocr_engine` + lifespan | `main.py` | 服务启动时就跑一次假推理，首个真实请求不用等 PaddlePaddle 冷启动 |

---

## 6. API 接口一览

> 详细字段说明、错误码、示例请看 [`API.md`](./API.md)。这里给出**速查表**。

| Method | Path | 主要入参 | 主要出参 | 典型耗时（GPU / CPU） |
|--------|------|----------|----------|-----------------------|
| GET | `/health` | — | `{"status": "ok"}` | < 1ms |
| POST | `/api/mask` | `image_url` (必填) / `llm_provider` / `mask_mode` / `categories` | `success / origin_url / masked_url / error` | 30–60s / 40–60s |
| POST | `/api/detect` | `image_url` / `llm_provider` / `categories` | `sensitive_items[] / total_lines / sensitive_count` | 30–50s / 35–55s |
| POST | `/api/ocr` | `image_url` | `lines[] / full_text / total_lines` | 2–3s / 10–15s |

**公共行为**：
- CORS 完全开放。
- 所有响应带 `X-Request-ID`。
- `/api/mask` 失败不抛 HTTP 错误码，改在响应体 `success=false, error=...`（便于前端一视同仁展示）。
- `/api/detect`、`/api/ocr` 按 REST 惯例返回 400 / 500。

**参数默认值**（代码里的真实值，以此为准）：

- `llm_provider` 不传 → 读 `DEFAULT_LLM_PROVIDER`（`.env` 默认 `volcengine-lite`）
- `mask_mode` 不传 → `"white"`
- `categories` 不传 → 9 类默认（见 5.4）

---

## 7. 配置项全集

`.env` 支持以下变量。所有项都有代码层默认值，`.env.example` 里的占位值仅供参考。

### 7.1 应用运行

| 变量 | 默认 | 说明 |
|------|------|------|
| `DEFAULT_LLM_PROVIDER` | `volcengine-lite` | `llm_provider` 未传时的 fallback |
| `MAX_CONCURRENCY` | `2` | 全局信号量 + 线程池上限 |
| `OCR_POOL_SIZE` | `0`（自动） | OCR 子进程数；0 = GPU 下 1 / CPU 下 `cpu_count - 1` |
| `OCR_MODEL` | `mobile` | `server`（精度高）/ `mobile`（速度快）/ `legacy`（回退老 API） |
| `APP_WORKERS` | `1` | uvicorn worker 数（建议保持 1，OCR 子进程池已并发） |

### 7.2 Vision 参数

| 变量 | 默认 | 说明 |
|------|------|------|
| `VISION_MAX_TOKENS` | `384` | LLM 响应上限 tokens |
| `VISION_IMAGE_MAX_SIDE` | `1280` | 送 LLM 的图片最长边（超过会等比缩放） |
| `VISION_JPEG_QUALITY` | `75` | 送 LLM 的 JPEG 编码质量 |

### 7.3 LLM 提供商

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_VISION_DETAIL` | OpenAI 兼容接口（`OPENAI_VISION_DETAIL` 可选 `auto / low / high`） |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` | Claude |
| `ZHIPU_API_KEY` / `ZHIPU_BASE_URL` / `ZHIPU_MODEL` | 智谱 GLM |
| `BAILIAN_API_KEY` / `BAILIAN_BASE_URL` / `BAILIAN_MODEL` | 阿里云百炼 Qwen Vision |
| `ARK_API_KEY` / `ARK_BASE_URL` / `ARK_MODEL` / `ARK_MODEL_LITE` | 火山引擎豆包（`ARK_MODEL_LITE` 给 `volcengine-lite` provider 用） |

### 7.4 腾讯云 COS

| 变量 | 默认 | 说明 |
|------|------|------|
| `COS_SECRET_ID` / `COS_SECRET_KEY` | — | 密钥 |
| `COS_REGION` | `ap-nanjing` | 默认区域 |
| `COS_BUCKET` | — | 默认桶（上传打码图时仍根据原图 URL 解析出的 bucket 输出） |
| `COS_SCHEME` | `https` | 生成 URL 的协议 |
| `COS_DOMAIN_MAP` | `""` | JSON 字符串，自定义域名映射，格式 `{"域名": "bucket/region"}` |

### 7.5 PaddlePaddle / 系统级

运行时在 `app/bootstrap/runtime.py` 统一设定（已在进程启动时兜底，`os.environ.setdefault` 不会覆盖用户显式值）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `FLAGS_enable_pir_api` | `0` | 关闭 PIR，规避部分兼容性问题 |
| `FLAGS_enable_pir_in_executor` | `0` | 同上 |
| `FLAGS_use_mkldnn` | `0` | 关闭 MKL-DNN（部分 CPU 下更稳） |
| `FLAGS_allocator_strategy` | `naive_best_fit` | 显存分配策略 |
| `FLAGS_fraction_of_gpu_memory_to_use` | `0.6` | GPU 占用上限 |
| `FLAGS_num_threads` | `1` | Paddle 内部线程 |
| `OMP_NUM_THREADS` | `2` | OpenMP 线程 |
| `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` | `1` | BLAS 线程 |
| `INSTALL_GPU` | `false` | 部署脚本使用；`true` 切到 GPU 模式 |
| `HF_ENDPOINT` / `HF_HUB_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 模型镜像源 |
| `FACE_YUNET_MODEL` | — | 自定义 YuNet ONNX 模型路径（默认使用仓库内模型） |

### 7.6 Docker 资源

| 变量 | 默认 | 说明 |
|------|------|------|
| `BASE_IMAGE` | `python:3.11-slim`（CPU）/ `nvidia/cuda:11.8.0-runtime-ubuntu22.04`（GPU） | Docker base image |
| `PADDLE_GPU_VERSION` | `3.1.1` | GPU 版 paddlepaddle 版本 |
| `PADDLE_CUDA_TAG` | `cu118` | CUDA 标签 |
| `DOCKER_CPUS` | `4` | 容器 CPU 限制 |
| `DOCKER_MEM` | `18G` | 容器内存限制 |
| `DOCKER_SHM_SIZE` | `2gb` | `/dev/shm` 大小（多进程 OCR 需要） |

---

## 8. 扩展点

项目通过 `Registry` 把三类核心组件全部插件化，新增扩展几乎不改老代码。

### 8.1 新增 LLM 检测器

```python
# app/core/detectors/myprovider.py
from .base import BaseDetector
from ..registry import detector_registry

@detector_registry.register("myprovider")
class MyProviderDetector(BaseDetector):
    def __init__(self):
        # 读取自己的 API Key、初始化 client ...
        pass

    def _call_vision(self, b64_data: str, media_type: str,
                     user_prompt: str, system_prompt: str) -> str:
        # 调自己的 Vision API，返回原始文本响应
        ...
```

然后在 `app/core/detectors/__init__.py` 加一行 `from .myprovider import MyProviderDetector  # noqa: F401`。
调用时传 `llm_provider=myprovider` 即可。**规则 + CV 兜底逻辑全部自动继承，无需重写。**

### 8.2 新增打码策略

```python
# app/core/masking/rainbow.py
import cv2, numpy as np
from .base import BaseMaskStrategy
from ..registry import masking_registry

@masking_registry.register("rainbow")
class RainbowStrategy(BaseMaskStrategy):
    def apply(self, image: np.ndarray, polygon: np.ndarray) -> None:
        # 原地修改 image
        ...
```

在 `app/core/masking/__init__.py` 里 `import` 触发注册。

### 8.3 新增 OCR 引擎

```python
# app/core/ocr/myocr.py
from .base import BaseOCREngine
from ..models import TextLine
from ..registry import ocr_registry

@ocr_registry.register("myocr")
class MyOCREngine(BaseOCREngine):
    def recognize(self, image_path: str) -> list[TextLine]:
        ...
```

### 8.4 新增业务接口

1. `app/api/v1/endpoints/` 下加新文件定义 `APIRouter`。
2. 在 `app/api/v1/__init__.py` 的 `v1_router.include_router(...)` 里加一行。
3. 若需 Pydantic schema，放 `app/schemas/` 并在包 `__init__.py` 导出。

### 8.5 新增敏感类别

1. 在 `app/core/detectors/base.py::CATEGORY_DESCRIPTIONS` 加条目。
2. 若需要 prompt 默认启用，把它加进 `DEFAULT_CATEGORIES`。
3. 如果该类别有固定字段特征，可在 `BaseDetector._extract_rule_based()` 加一条正则，和 LLM 结果一起进 dedup。

---

## 9. 性能与运维

### 9.1 典型耗时构成（单张图，GPU 环境）

```
方向矫正       ~10–30ms
OCR 首次       ~1–3s   （server 模型）/ ~0.5–1.5s（mobile 模型）
OCR 重试       ~1–3s   （仅竖排/乱码时发生）
Vision LLM     ~10–40s （主瓶颈，各 provider 差异大）
CV 手写/人脸   ~50–200ms
遮罩 + 编码    ~100–300ms
──────────────
合计           ~15–45s（正常）/ ~2–5s（无文字单据直出）
```

**瓶颈始终在 Vision LLM**，GPU 主要加速 OCR 阶段，对总耗时影响约 1–2s。

### 9.2 80 并发推荐配置

- **GPU 方案**：8 核 32G + T4 16G，`MAX_CONCURRENCY=80, OCR_POOL_SIZE=4, OCR_MODEL=server`。
- **CPU 方案**：16 核 64G，`MAX_CONCURRENCY=80, OCR_POOL_SIZE=16, OCR_MODEL=server`。
- 详细参见 `README.md` 的"服务器选购指南"。

### 9.3 日志与可观测

每次 mask 请求会输出：
- `[MASK]` 前缀：请求入口 / STEP 1–3 / 上传 URL / 总耗时。
- `[Pipeline]` 前缀：方向矫正、OCR、重试、LLM 检测、遮罩各阶段 ms 级耗时。
- `[PERF]` 前缀：同上，额外 stderr 镜像输出，便于 `tail -f`。
- `[Detect]` 前缀：detect 接口的简化耗时日志。

每次响应都带 `X-Request-ID`，8 位十六进制，全链路日志都以此 ID 前缀对齐。

### 9.4 常见故障与恢复

| 现象 | 原因 | 自动恢复 | 需要人工介入吗 |
|------|------|----------|-----------------|
| OCR worker 崩溃（exit code != 0） | PaddlePaddle SIGSEGV / OOM | ✅ 子进程管理器自动重启 | 否 |
| `当前系统压力过大，请稍后重试` | OCR OOM / 子进程超时 / cublas 报错 | ✅ 子进程重启 + 返回业务错误 | 若频繁，调小 `MAX_CONCURRENCY` 或升配 |
| Vision LLM 超时 | 300s 内 LLM 无响应 | ❌ 抛 500 `脱敏处理失败，请稍后重试` | 检查 API Key / 切换 provider |
| `不支持的文件格式` | 魔数 / 扩展名不匹配 | — | 前端传对图片 |
| `无法解析 COS URL` | 非 myqcloud 域名且未配 `COS_DOMAIN_MAP` | — | 补配 COS_DOMAIN_MAP |

---

## 10. 已知边界与限制

- **整页纯手写**：当前手写签名检测面向"印刷单据 + 零散手写字段"设计，整页全手写（如手写病历）命中率会下降，需要依赖 LLM 的 `handwritten_signature` 判定，且 CV 二次校验可能误伤。
- **复杂版面**：多栏、嵌入表格的单据，OCR 行序可能和视觉阅读顺序不符，LLM 偶尔会给出错误的 `line_index`，服务端有"前缀容错 / 跨行模糊匹配"兜底，仍可能漏 1–2 个。
- **非中文文档**：PaddleOCR 采用 `lang='ch'`，英文纯文本单据识别可用但精度不如中文；未做多语言切换。
- **图像质量**：极度模糊（< 300px 短边）或严重倾斜（> 30°）场景未专项优化，建议上传前在客户端做一次基本校准。
- **打码可逆性**：当前输出是**不可逆**的覆盖/模糊，没有密文映射功能；若需可恢复的 tokenization，需要在业务层自己维护映射表。
- **MAX_CONCURRENCY 与真实吞吐**：`MAX_CONCURRENCY` 只是并发入口数，真实吞吐上限在 Vision LLM API 的 QPS 与 OCR 子进程数之间；盲目调大会导致 LLM 侧 429 / 超时。
- **`.env.example` 含真实密钥**（注意）：当前 `.env.example` 中 LLM / COS 密钥为真实值，**不应发布到公开仓库**；部署时请务必替换或从 secrets 注入。

---

## 附：目录结构速查

```
medical_tool/
├── run.py                              # uvicorn 启动入口
├── requirements.txt
├── Dockerfile + docker-compose{,.gpu}.yml
├── deploy.sh / deploy-bare.sh
├── .env.example
├── README.md                           # 部署 + 快速上手
├── API.md                              # 接口字段细节
├── FEATURES.md                         # 本文：功能全景
├── models/
│   └── face_detection_yunet.onnx       # 人脸检测模型
└── app/
    ├── main.py                         # FastAPI app + lifespan + /health
    ├── config.py                       # Pydantic Settings
    ├── bootstrap/runtime.py            # 启动前环境变量注入
    ├── api/v1/
    │   ├── __init__.py
    │   ├── deps.py                     # 组件工厂 + 信号量 + 线程池 + 预热锁
    │   └── endpoints/
    │       ├── mask.py                 # POST /api/mask
    │       ├── detect.py               # POST /api/detect
    │       └── ocr.py                  # POST /api/ocr
    ├── core/
    │   ├── pipeline.py                 # MedicalPrivacyMasker + OrientationCorrector
    │   ├── registry.py                 # 插件注册表
    │   ├── models.py                   # TextLine / SensitiveRegion / ProcessStats
    │   ├── errors.py                   # SystemOverloadError
    │   ├── cos.py                      # 腾讯云 COS 封装
    │   ├── face_detector.py            # YuNet / Haar 人脸检测
    │   ├── ocr/                        # PaddleOCR 子进程池
    │   ├── detectors/                  # 5 家 Vision LLM + base 规则/CV 兜底
    │   └── masking/                    # white / black / gray / blur / mosaic
    ├── schemas/                        # Pydantic 请求/响应
    ├── middleware/error_handler.py     # CORS + 请求日志 + 全局异常
    └── utils/image.py                  # 图像读写 / base64 / 多边形计算
```