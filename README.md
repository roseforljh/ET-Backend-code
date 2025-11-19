---
title: EzTalk Proxy
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
# 关键：添加以下两行以确保网络访问和密钥可用
network: true
secrets:
  - GOOGLE_API_KEY
  - ADMIN_PASSWORD
  - SIGNATURE_SECRET_KEYS
---

# EzTalk Proxy Service

EzTalk Proxy 是一个高性能、安全的 FastAPI 后端服务，专为 EzTalk 客户端设计。它作为统一的 API 网关，处理与各种大型语言模型（如 Google Gemini、OpenAI 兼容接口）的通信，并提供图像生成、语音对话等扩展功能。

## ✨ 主要特性

*   **多模型支持**: 统一封装 Google Gemini 和 OpenAI 兼容格式的 API。
*   **智能路由**: 根据请求参数自动分发到直连渠道或聚合商渠道。
*   **流式响应**: 完美支持 SSE (Server-Sent Events) 流式输出，打字机效果流畅。
*   **安全机制**:
    *   **请求签名验证**: 防止未授权的接口调用。
    *   **管理后台**: 内置可视化管理面板，监控系统状态。
*   **扩展功能**:
    *   **Gemini Live**: 支持实时语音对话中转。
    *   **图像生成**: 集成多种绘图模型接口。
    *   **联网搜索**: 支持集成 Google Search 等搜索服务。
*   **可观测性**: 详细的访问日志记录和统计图表。

## 🚀 快速开始

### 本地开发

1.  **克隆仓库**
    ```bash
    git clone https://github.com/your-repo/eztalk-proxy.git
    cd eztalk-proxy
    ```

2.  **创建虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量**
    复制 `.env.example` 为 `.env` 并填写配置：
    ```bash
    cp .env.example .env
    ```
    主要配置项：
    *   `GOOGLE_API_KEY`: Google AI Studio API 密钥。
    *   `ADMIN_PASSWORD`: 管理后台登录密码。
    *   `SIGNATURE_VERIFICATION_ENABLED`: 是否开启请求签名验证 (true/false)。

5.  **启动服务**
    ```bash
    python run.py
    # 或者直接使用 uvicorn
    uvicorn eztalk_proxy.main:app --host 0.0.0.0 --port 8000 --reload
    ```

6.  **访问服务**
    *   API 文档: `http://localhost:8000/docs`
    *   管理后台: `http://localhost:8000/everytalk`

### 🐳 Docker 部署

```bash
# 构建镜像
docker build -t eztalk-proxy .

# 运行容器
docker run -d -p 7860:7860 \
  -e GOOGLE_API_KEY=your_key \
  -e ADMIN_PASSWORD=your_password \
  eztalk-proxy
```

使用 Docker Compose (推荐):
```bash
docker-compose up -d
```

## 🛠️ 环境变量说明

| 变量名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `PORT` | 服务监听端口 | `7860` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `ADMIN_PASSWORD` | 管理后台登录密码 | `admin` |
| `GOOGLE_API_KEY` | Gemini API 密钥 | - |
| `SIGNATURE_VERIFICATION_ENABLED` | 是否启用请求签名验证 | `false` |
| `SIGNATURE_SECRET_KEYS` | 签名密钥列表(逗号分隔) | - |
| `MAX_CONNECTIONS` | 最大并发连接数 | `100` |
| `API_TIMEOUT` | API 请求超时时间(秒) | `120` |

## 📊 管理后台

访问 `/everytalk` 路径即可进入管理后台。

*   **仪表盘**: 查看今日访问量、系统运行时间、资源占用情况及访问趋势图。
*   **访问日志**: 查看详细的 API 调用记录（支持按时间、IP、路径筛选）。
*   **系统配置**: 在线查看和修改环境变量配置。
*   **安全中心**: 修改管理员密码。

> **注意**: 统计数据已排除本地测试 IP (127.0.0.1, ::1, localhost)。

## ☁️ 部署到 Hugging Face Spaces

本项目已针对 Hugging Face Spaces (Docker SDK) 进行了优化。

1.  创建一个新的 Space，SDK 选择 **Docker**。
2.  上传代码到 Space 的仓库。
3.  在 Space 的 **Settings** -> **Repository secrets** 中添加必要的环境变量：
    *   `GOOGLE_API_KEY`
    *   `ADMIN_PASSWORD`
    *   其他需要的配置...
4.  等待构建完成即可使用。

## 📝 License

MIT