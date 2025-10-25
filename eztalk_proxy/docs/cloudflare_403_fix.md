# Cloudflare 403 错误修复指南

## 问题现象

当使用第三方 Gemini API 代理（如 `gemini.jhun.edu.kg`）时，可能会遇到 403 错误，错误信息显示 Cloudflare 的质询页面（"Just a moment..."）或返回乱码数据。

## 问题原因

1. **Cloudflare 反爬虫机制**：代理服务器使用 Cloudflare 作为防护，会检测请求是否来自真实浏览器
2. **缺少浏览器特征**：Python httpx 默认的请求头看起来像脚本请求，触发了安全检查
3. **压缩问题**：某些请求头组合可能导致响应被压缩但未正确解压

## 已实施的修复

### 1. 添加浏览器特征请求头

代码已更新为在访问第三方代理时自动添加完整的浏览器请求头：

```python
# 对于第三方代理，添加：
- User-Agent: Chrome 浏览器标识
- Origin: 代理域名
- Referer: 代理主页
- Sec-Ch-Ua: 浏览器版本信息
- Sec-Fetch-* 系列：浏览器安全策略头
- Accept-Language: 语言偏好
```

### 2. 自动压缩处理

- 移除了手动设置的 `Accept-Encoding` 头
- 让 httpx 自动处理 gzip/br 压缩和解压
- 避免乱码问题

### 3. 智能判断

- 对 Google 官方 API（`googleapis.com`）：使用标准头
- 对第三方代理：添加完整浏览器特征头

## 如果仍然失败

### 方案 1：更换代理服务

建议使用以下更可靠的代理服务：
- `api.aiproxy.io`
- `api.chatanywhere.com.cn`
- 或申请官方 Google AI Studio API Key

### 方案 2：检查代理要求

某些代理可能需要：
- 特殊的 API Key 格式
- 额外的认证头
- 白名单 IP
- 特定的请求频率限制

### 方案 3：使用 OpenAI 兼容格式

如果代理同时支持 OpenAI 兼容接口，可以尝试：
1. 在 App 设置中选择 "OpenAI 兼容"
2. 填入代理的 OpenAI 兼容端点（如 `/v1/chat/completions`）
3. 这样可以绕过 Gemini 原生 API 的 Cloudflare 检查

### 方案 4：联系代理服务商

如果是付费代理服务：
- 询问是否有 IP 白名单机制
- 要求提供正确的请求头配置
- 确认服务是否正常运行

## 技术细节

### 修改的文件

1. `backdAiTalk/eztalk_proxy/services/requests/builders/gemini_builder.py`
   - 第 219-249 行：添加浏览器请求头逻辑

2. `backdAiTalk/eztalk_proxy/services/requests/headers.py`
   - 第 28-40 行：更新 OpenAI 兼容接口的请求头

### 为什么有些人不报错？

可能的原因：
1. **IP 信誉**：你的服务器 IP 可能被标记为可疑
2. **请求频率**：请求过于频繁触发了限流
3. **地理位置**：某些代理可能有地域限制
4. **客户端差异**：别人可能用的是不同的客户端库或有特殊配置

### 调试建议

1. **查看完整日志**：
   ```bash
   # 查看请求头
   tail -f logs/app.log | grep "Added browser-like headers"
   ```

2. **测试代理连通性**：
   ```bash
   curl -v https://gemini.jhun.edu.kg/v1beta/models/gemini-2.5-flash:streamGenerateContent?key=YOUR_KEY \
     -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
     -H "Origin: https://gemini.jhun.edu.kg" \
     -H "Referer: https://gemini.jhun.edu.kg/"
   ```

3. **尝试直接访问**：
   用浏览器打开 `https://gemini.jhun.edu.kg`，看是否能正常访问

## 更新日志

- **2025-01-20**: 初始修复，添加浏览器请求头
- **2025-01-20**: 移除 Accept-Encoding，修复乱码问题

