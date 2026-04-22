# 医疗单据隐私脱敏 API 文档

> **Base URL:** `http://<host>:8000`
>
> **框架:** FastAPI | **版本:** 1.0.0

---

## 目录

- [1. 健康检查](#1-健康检查)
- [2. 图片打码](#2-图片打码)
- [3. 敏感信息检测](#3-敏感信息检测)
- [4. OCR 文字提取](#4-ocr-文字提取)
- [附录 A：敏感类别说明](#附录-a敏感类别说明)
- [附录 B：打码模式说明](#附录-b打码模式说明)
- [附录 C：环境变量配置](#附录-c环境变量配置)
- [附录 D：错误处理](#附录-d错误处理)

---

## 1. 健康检查

检查服务是否正常运行。

```
GET /health
```

### 请求示例

```bash
curl http://localhost:8000/health
```

### 响应示例

```json
{
  "status": "ok"
}
```

---

## 2. 图片打码

从腾讯云 COS 下载医疗单据图片，自动检测并遮盖敏感信息，保留原图 URL 不变，并将打码后的图片上传为新的 `masked_url`。

```
POST /api/mask
```

### 请求参数

| 字段           | 类型       | 必填 | 默认值     | 说明                                                         |
| -------------- | ---------- | ---- | ---------- | ------------------------------------------------------------ |
| `image_url`    | `string`   | 是   | —          | 腾讯云 COS 图片 URL                                         |
| `llm_provider` | `string`   | 否   | `null`     | 视觉大模型供应商；为空时使用服务端 `DEFAULT_LLM_PROVIDER`，支持 `openai` / `claude` / `zhipu` / `bailian` / `volcengine` / `volcengine-lite` |
| `mask_mode`    | `string`   | 否   | `"white"`  | 打码模式，可选 `"black"` / `"white"` / `"gray"` / `"blur"` / `"mosaic"` |
| `categories`   | `string[]` | 否   | `null`     | 需要检测的敏感类别列表，`null` 时使用默认类别                |

### 请求示例

**基础用法 — 使用所有默认参数：**

```bash
curl -X POST http://localhost:8000/api/mask \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001.jpg"
  }'
```

**指定 Claude 作为检测模型 + 马赛克打码：**

```bash
curl -X POST http://localhost:8000/api/mask \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001.jpg",
    "llm_provider": "claude",
    "mask_mode": "mosaic"
  }'
```

**仅检测姓名和身份证号：**

```bash
curl -X POST http://localhost:8000/api/mask \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001.jpg",
    "categories": ["name", "id_card"]
  }'
```

**使用高斯模糊 + 检测所有常见类别：**

```bash
curl -X POST http://localhost:8000/api/mask \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001.png",
    "llm_provider": "openai",
    "mask_mode": "blur",
    "categories": ["name", "id_card", "phone", "medical_id", "doctor", "email", "address", "age", "date_of_birth", "hospital"]
  }'
```

### 响应参数

| 字段         | 类型      | 说明                              |
| ------------ | --------- | --------------------------------- |
| `success`    | `boolean` | 处理是否成功                      |
| `origin_url` | `string?` | 原图 URL（成功时与请求 `image_url` 一致） |
| `masked_url` | `string?` | 打码后图片的新 COS URL（失败时为 null） |
| `error`      | `string?` | 错误信息（成功时为 null）          |

### 成功响应示例

```json
{
  "success": true,
  "origin_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001.jpg",
  "masked_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001_masked_20260323143025.jpg",
  "error": null
}
```

### 失败响应示例

```json
{
  "success": false,
  "origin_url": null,
  "masked_url": null,
  "error": "无效的 mask_mode，可选: ['black', 'gray', 'white', 'blur', 'mosaic']"
}
```

```json
{
  "success": false,
  "origin_url": null,
  "masked_url": null,
  "error": "不支持的文件格式，仅支持: .jpg, .jpeg, .png, .bmp, .webp, .tiff, .tif"
}
```

### 处理流程

```
1. 从 COS 下载原图
2. OCR 识别 → Vision LLM 检测敏感区域 → 应用打码
3. 将打码图片上传到新的 masked URL
4. 返回 origin_url 和 masked_url
```

---

## 3. 敏感信息检测

仅检测医疗单据图片中的敏感信息，**不进行打码处理**，返回检测结果。

```
POST /api/detect
```

### 请求参数

| 字段           | 类型       | 必填 | 默认值     | 说明                                                    |
| -------------- | ---------- | ---- | ---------- | ------------------------------------------------------- |
| `image_url`    | `string`   | 是   | —          | 腾讯云 COS 图片 URL                                    |
| `llm_provider` | `string`   | 否   | `null`     | 视觉大模型供应商；为空时使用服务端 `DEFAULT_LLM_PROVIDER`，支持 `openai` / `claude` / `zhipu` / `bailian` / `volcengine` / `volcengine-lite` |
| `categories`   | `string[]` | 否   | `null`     | 需要检测的敏感类别列表，`null` 时使用默认类别           |

### 请求示例

**使用默认参数检测所有默认类别：**

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001.jpg"
  }'
```

**使用 Claude 检测，仅查找手机号和身份证：**

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001.jpg",
    "llm_provider": "claude",
    "categories": ["phone", "id_card"]
  }'
```

### 响应参数

| 字段              | 类型              | 说明                 |
| ----------------- | ----------------- | -------------------- |
| `sensitive_items`  | `SensitiveItem[]` | 检测到的敏感信息列表 |
| `total_lines`     | `int`             | OCR 识别的总行数     |
| `sensitive_count` | `int`             | 敏感信息条数         |

**SensitiveItem 结构：**

| 字段             | 类型                     | 说明                              |
| ---------------- | ------------------------ | --------------------------------- |
| `text`           | `string`                 | 敏感文本内容                      |
| `category`       | `string`                 | 敏感类别（见附录 A）              |
| `bbox`           | `[int, int, int, int]`   | 边界框坐标 `[x1, y1, x2, y2]`    |
| `ocr_line_index` | `int`                    | 所在 OCR 行的索引                 |

### 成功响应示例

```json
{
  "sensitive_items": [
    {
      "text": "张三",
      "category": "name",
      "bbox": [120, 85, 210, 115],
      "ocr_line_index": 2
    },
    {
      "text": "320102199001011234",
      "category": "id_card",
      "bbox": [300, 85, 580, 115],
      "ocr_line_index": 3
    },
    {
      "text": "13800138000",
      "category": "phone",
      "bbox": [120, 140, 310, 170],
      "ocr_line_index": 5
    },
    {
      "text": "李医生",
      "category": "doctor",
      "bbox": [400, 620, 490, 650],
      "ocr_line_index": 18
    }
  ],
  "total_lines": 25,
  "sensitive_count": 4
}
```

### 错误响应示例

- **400 Bad Request** — URL 格式错误或文件格式不支持

```json
{
  "detail": "不支持的文件格式，仅支持: .jpg, .jpeg, .png, .bmp, .webp, .tiff, .tif"
}
```

- **500 Internal Server Error** — 检测过程异常

```json
{
  "detail": "检测失败: ..."
}
```

---

## 4. OCR 文字提取

提取医疗单据图片中的全部文字内容（基于 PaddleOCR），返回逐行结果。

```
POST /api/ocr
```

### 请求参数

| 字段        | 类型     | 必填 | 说明               |
| ----------- | -------- | ---- | ------------------ |
| `image_url` | `string` | 是   | 腾讯云 COS 图片 URL |

### 请求示例

```bash
curl -X POST http://localhost:8000/api/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/reports/patient_001.jpg"
  }'
```

### 响应参数

| 字段          | 类型            | 说明                               |
| ------------- | --------------- | ---------------------------------- |
| `lines`       | `OcrTextLine[]` | 每行文字的详细信息                 |
| `full_text`   | `string`        | 全部文字拼接（以换行符分隔）       |
| `total_lines` | `int`           | 识别出的总行数                     |

**OcrTextLine 结构：**

| 字段         | 类型                   | 说明                            |
| ------------ | ---------------------- | ------------------------------- |
| `index`      | `int`                  | 行序号（从 0 开始）             |
| `text`       | `string`               | 该行识别出的文本                |
| `bbox`       | `[int, int, int, int]` | 边界框坐标 `[x1, y1, x2, y2]`  |
| `confidence` | `float`                | OCR 置信度（0 ~ 1）            |

### 成功响应示例

```json
{
  "lines": [
    {
      "index": 0,
      "text": "XX市第一人民医院",
      "bbox": [150, 20, 450, 55],
      "confidence": 0.98
    },
    {
      "index": 1,
      "text": "门诊病历",
      "bbox": [230, 60, 370, 90],
      "confidence": 0.99
    },
    {
      "index": 2,
      "text": "姓名：张三",
      "bbox": [50, 100, 200, 130],
      "confidence": 0.95
    },
    {
      "index": 3,
      "text": "身份证号：320102199001011234",
      "bbox": [250, 100, 580, 130],
      "confidence": 0.93
    }
  ],
  "full_text": "XX市第一人民医院\n门诊病历\n姓名：张三\n身份证号：320102199001011234",
  "total_lines": 4
}
```

### 错误响应示例

- **400 Bad Request**

```json
{
  "detail": "不支持的文件格式，仅支持: .jpg, .jpeg, .png, .bmp, .webp, .tiff, .tif"
}
```

- **500 Internal Server Error**

```json
{
  "detail": "OCR 识别失败: ..."
}
```

---

## 附录 A：敏感类别说明

### 默认类别（`categories` 为 `null` 时自动使用）

| 类别值       | 说明                                                                                 |
| ------------ | ------------------------------------------------------------------------------------ |
| `name`       | 患者/就诊人真实姓名（大部分 2-4 个汉字，少数民族可能更长）                           |
| `id_card`    | 身份证号码（15 位或 18 位数字）                                                       |
| `phone`      | 手机或固定电话号码                                                                    |
| `medical_id` | 病历号/就诊号/住院号/超声号/门诊号/检查号等各类编号                                   |
| `doctor`     | 医疗单据上的医生/医务人员姓名（申请医生、报告医生、审核医师、检查医生、主治医师等）   |
| `email`      | 电子邮箱地址（如 xxx@xxx.com）                                                        |
| `address`    | 家庭住址/居住地址/户籍地址/通讯地址等个人地址信息                                     |

### 可选扩展类别（需显式传入 `categories` 数组）

| 类别值              | 说明                       |
| ------------------- | -------------------------- |
| `age`               | 年龄数值（如 "33岁"）     |
| `date_of_birth`     | 出生日期                   |
| `date`              | 所有日期格式内容（检查日期、报告日期、入院日期等） |
| `hospital`          | 医院全称或简称（含分院、院区、单据顶部医院标题）   |
| `portrait`          | 真人人像照片区域（证件照、头像等，由 CV 自动识别） |

---

## 附录 B：打码模式说明

| 模式值    | 效果                                           |
| --------- | ---------------------------------------------- |
| `black`   | 纯黑色填充覆盖（RGB: 0, 0, 0）                |
| `gray`    | 纯灰色填充覆盖（RGB: 128, 128, 128）           |
| `white`   | 纯白色填充覆盖（RGB: 255, 255, 255）           |
| `blur`    | 高斯模糊，模糊核大小自适应区域尺寸             |
| `mosaic`  | 马赛克效果（15px 像素块）                       |

---

## 附录 C：环境变量配置

通过 `.env` 文件或环境变量配置，所有参数均有默认值。

| 变量名                | 默认值                       | 说明                        |
| --------------------- | ---------------------------- | --------------------------- |
| `OPENAI_API_KEY`       | `""`                           | OpenAI API 密钥                               |
| `OPENAI_BASE_URL`      | `""`                           | OpenAI 兼容接口 Base URL，可留空使用官方地址   |
| `OPENAI_MODEL`         | `"gpt-4o"`                     | OpenAI 模型名称                               |
| `OPENAI_VISION_DETAIL` | `"auto"`                       | OpenAI 图片细节级别                           |
| `ANTHROPIC_API_KEY`    | `""`                           | Anthropic API 密钥                            |
| `ANTHROPIC_BASE_URL`   | `""`                           | Anthropic 自定义 Base URL                     |
| `ANTHROPIC_MODEL`      | `"claude-sonnet-4-20250514"`   | Anthropic 模型名称                            |
| `ZHIPU_API_KEY`        | `""`                           | 智谱 API 密钥                                 |
| `ZHIPU_BASE_URL`       | `"https://open.bigmodel.cn/api/paas/v4/"` | 智谱 OpenAI 兼容接口 Base URL       |
| `ZHIPU_MODEL`          | `"GLM-4.6V-Flash"`             | 智谱模型名称；当前官方文档标注为免费模型，适合初步跑通测试 |
| `BAILIAN_API_KEY`      | `""`                           | 阿里云百炼 API 密钥                           |
| `BAILIAN_BASE_URL`     | `"https://dashscope.aliyuncs.com/compatible-mode/v1"` | 百炼兼容接口 Base URL |
| `BAILIAN_MODEL`        | `"qwen3-vl-32b-instruct"`      | 百炼模型名称                                  |
| `ARK_API_KEY`          | `""`                           | 火山引擎豆包 API 密钥                         |
| `ARK_BASE_URL`         | `"https://ark.cn-beijing.volces.com/api/v3"` | 火山引擎兼容接口 Base URL         |
| `ARK_MODEL`            | `"doubao-seed-1-8-251228"`     | 火山引擎主模型名称                            |
| `ARK_MODEL_LITE`       | `"doubao-seed-2-0-lite-260215"` | 火山引擎轻量模型名称                        |
| `DEFAULT_LLM_PROVIDER` | `"volcengine-lite"`            | 默认 LLM 供应商                               |
| `COS_SECRET_ID`        | `""`                           | 腾讯云 COS SecretId                           |
| `COS_SECRET_KEY`       | `""`                           | 腾讯云 COS SecretKey                          |
| `COS_REGION`           | `"ap-nanjing"`                 | COS 区域                                      |
| `COS_BUCKET`           | `""`                           | COS 存储桶名称                                |
| `COS_SCHEME`           | `"https"`                      | COS 协议                                      |
| `COS_DOMAIN_MAP`       | `""`                           | 自定义域名映射 JSON，格式 `{"cdn.example.com":"bucket/region"}` |
| `MAX_CONCURRENCY`      | `2`                            | 最大并发处理数                                |
| `OCR_MODEL`            | `"mobile"`                     | OCR 模型类型                                  |
| `OCR_POOL_SIZE`        | `0`                            | OCR 实例池大小，`0` 表示自动                   |
| `VISION_MAX_TOKENS`    | `384`                          | Vision LLM 最大输出 token 数                  |
| `VISION_IMAGE_MAX_SIDE`| `1280`                         | 发送给 Vision LLM 的图片最大边长              |
| `VISION_JPEG_QUALITY`  | `75`                           | Vision 图片压缩质量                           |

---

## 附录 D：错误处理

### 通用行为

- 所有请求均记录日志（方法、路径、状态码、耗时）
- 响应头包含 `X-Request-ID` 用于追踪
- CORS 已开放（允许所有来源和方法）

### `/api/mask` 错误处理

该接口在响应体中返回错误，HTTP 状态码始终为 **200**：

```json
{
  "success": false,
  "origin_url": null,
  "masked_url": null,
  "error": "具体错误信息"
}
```

常见错误：
- `"无效的 mask_mode，可选: ['black', 'gray', 'white', 'blur', 'mosaic']"`
- `"不支持的文件格式，仅支持: .jpg, .jpeg, .png, .bmp, .webp, .tiff, .tif"`
- `"图片下载失败: ..."`
- `"脱敏处理失败: ..."`
- `"上传打码图片失败: ..."`

### `/api/detect` 和 `/api/ocr` 错误处理

这两个接口使用标准 HTTP 状态码：

| 状态码 | 场景                               |
| ------ | ---------------------------------- |
| 400    | URL 格式错误、文件格式不支持       |
| 500    | 检测/识别过程异常                  |

### 支持的图片格式

`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`、`.tiff`、`.tif`

### `image_url` 格式要求

必须为合法的腾讯云 COS URL，格式：

```
https://<bucket>.cos.<region>.myqcloud.com/<key>
```

示例：

```
https://your-bucket-1250000000.cos.ap-nanjing.myqcloud.com/medical/report_001.jpg
```
