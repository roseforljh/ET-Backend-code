# Cloudflare 403 错误 - 完整解决方案

## 问题原因

你的后端服务器（Render/Zeabur/ClawCloud）的 IP 被 Cloudflare 识别为**云服务器/数据中心 IP**，触发了严格的机器人检测：

```
正常流程：
手机 (家庭网络 IP ✓)
  → 后端跳板服务器 (云服务器 IP ✗ Cloudflare拦截!)  
  → gemini.jhun.edu.kg

为什么会被拦截？
- Cloudflare 要求执行 JavaScript 质询（"Just a moment..."）
- Python httpx 无法执行 JavaScript
- 云服务器 IP 被标记为高风险
```

## 解决方案对比

### 方案 1：更换 Gemini API 地址（最简单 ⭐推荐）

**不需要修改任何代码**，只需在 App 设置中更换 API 地址！

#### 推荐的 API 地址：

1. **Google 官方 API**（最稳定）
   ```
   - 渠道：选择 "Gemini 官方"
   - API 地址：留空（或填 https://generativelanguage.googleapis.com）
   - API Key：从 https://aistudio.google.com 申请
   - 说明：需要科学上网，但非常稳定
   ```

2. **AIProxy**（国内可用）
   ```
   - 渠道：选择 "Gemini 官方"
   - API 地址：https://api.aiproxy.io
   - API Key：注册 AIProxy 获取
   - 说明：国内可直连，有免费额度
   ```

3. **使用 OpenAI 兼容格式**
   ```
   - 渠道：选择 "OpenAI 兼容"
   - API 地址：https://api.aiproxy.io/v1
   - API Key：你的 AIProxy key
   - 模型：gemini-2.0-flash-exp
   - 说明：通过 OpenAI 格式调用，绕过 Cloudflare
   ```

### 方案 2：添加"直连模式"开关（需要修改代码）

这就是别人给你提议的"开关"！

#### 工作原理：

```
跳板模式（现有）：
手机 → 后端跳板 (被Cloudflare拦) → Gemini API  ✗

直连模式（新增）：
手机 (家庭网络IP) → 直接连接 → Gemini API  ✓
```

#### 为什么直连能解决问题？

- ✅ 手机使用的是**家庭宽带/移动网络 IP**，Cloudflare 认为是正常用户
- ✅ 绕过了云服务器 IP 的检测
- ✅ App 可以直接与 Gemini API 通信

#### 需要修改的地方：

1. **在 App 设置中添加开关选项**
   ```kotlin
   // SettingsScreen.kt 中添加
   var useDirectMode by remember { mutableStateOf(false) }
   
   SwitchSetting(
       title = "直连模式",
       description = "绕过后端服务器，直接连接 Gemini API（可避免 Cloudflare 拦截）",
       checked = useDirectMode,
       onCheckedChange = { useDirectMode = it }
   )
   ```

2. **在 ApiClient 中添加直连逻辑**
   ```kotlin
   // ApiClient.kt
   suspend fun sendChatMessage(
       request: ChatRequest,
       useDirectMode: Boolean  // 新增参数
   ): Flow<AppStreamEvent> {
       return if (useDirectMode) {
           // 直连模式：App 直接请求 Gemini API
           sendDirectToGemini(request)
       } else {
           // 跳板模式：经过后端服务器
           sendViaBackend(request)
       }
   }
   ```

3. **实现直连函数**
   ```kotlin
   private suspend fun sendDirectToGemini(
       request: ChatRequest
   ): Flow<AppStreamEvent> = flow {
       val geminiUrl = request.api_address ?: "https://generativelanguage.googleapis.com"
       val model = request.model
       val url = "$geminiUrl/v1beta/models/$model:streamGenerateContent?key=${request.api_key}&alt=sse"
       
       // 构造 Gemini API 请求体
       val payload = buildGeminiPayload(request)
       
       // 发送请求
       val response = httpClient.post(url) {
           contentType(ContentType.Application.Json)
           setBody(payload)
       }
       
       // 解析 SSE 流
       parseGeminiSSE(response.bodyAsChannel())
   }
   ```

### 方案 3：配置请求头策略（已实现）

后端已经添加了 `CLOUDFLARE_BYPASS_STRATEGY` 配置，但**对于严格的 JS 质询无效**。

可以在环境变量中设置：
```bash
# .env 文件
CLOUDFLARE_BYPASS_STRATEGY=full   # full/minimal/none
```

## 最终建议

### 短期解决方案（立即可用）：
1. **更换 API 地址**到 `api.aiproxy.io` 或官方地址
2. 或使用 OpenAI 兼容格式

### 长期解决方案（最佳体验）：
1. **添加直连模式开关**
2. 让用户选择：
   - 跳板模式：便于监控和管理
   - 直连模式：避免 Cloudflare 拦截

## 为什么别人能用？

1. **使用不同的后端服务器**
   - 他们的服务器 IP 可能还没被 Cloudflare 标记

2. **使用不同的 API 地址**
   - 可能用的是官方 API 或其他代理

3. **使用直连模式**
   - 他们的 App 可能已经支持直连

4. **地理位置差异**
   - Cloudflare 对不同地区有不同的策略

## 技术细节

### Cloudflare 如何检测云服务器？

1. **IP 段检测**
   - 维护已知云服务商的 IP 列表
   - AWS、GCP、Azure、DigitalOcean 等

2. **TLS 指纹**
   - Python httpx 的 TLS 握手特征
   - 与真实浏览器不同

3. **请求模式**
   - 请求频率
   - User-Agent 特征
   - 缺少浏览器特有的头

### 为什么添加请求头无效？

因为 Cloudflare 使用了 **JavaScript 质询**：
```html
<script>
  // 计算质询 token
  window._cf_chl_opt = {...};
  // 需要执行 JS 才能通过
</script>
```

Python/Kotlin HTTP 客户端**无法执行 JavaScript**，所以无法通过质询。

## 总结

| 方案 | 难度 | 效果 | 推荐度 |
|------|------|------|--------|
| 更换 API 地址 | ⭐ 简单 | ✅ 立即解决 | ⭐⭐⭐⭐⭐ |
| 添加直连模式 | ⭐⭐⭐ 中等 | ✅ 完美解决 | ⭐⭐⭐⭐ |
| 修改请求头 | ⭐⭐ 简单 | ❌ 对 JS 质询无效 | ⭐ |

**立即行动**：先更换 API 地址到 `api.aiproxy.io`，然后考虑添加直连模式开关作为长期方案。

