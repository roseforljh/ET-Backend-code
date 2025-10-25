# 代码执行功能实现计划

## 概述
根据 Gemini 官方文档实现代码执行工具支持，允许模型生成和运行 Python 代码。

## 实现策略
- **启用策略**: 根据用户问题智能判断（auto模式），或显式开启/关闭
- **支持渠道**: Gemini REST 原生 + OpenAI 兼容格式
- **流式输出**: 独立事件传输代码片段和执行结果，不混入正文

## 改动清单

### 1. 数据模型扩展 (api_models.py)

#### 1.1 ChatRequestModel 新增字段
```python
enable_code_execution: Optional[bool] = Field(None, alias="enableCodeExecution")
# None = auto (智能判断), True = 强制开启, False = 强制关闭
```

#### 1.2 AppStreamEventPy 新增字段
```python
# 代码执行相关事件
executable_code: Optional[str] = Field(None, alias="executableCode")  # 生成的代码
code_language: Optional[str] = Field(None, alias="codeLanguage")  # 默认 "python"
code_execution_output: Optional[str] = Field(None, alias="codeExecutionOutput")  # 执行结果输出
code_execution_outcome: Optional[str] = Field(None, alias="codeExecutionOutcome")  # "success" | "error"
```

### 2. Gemini REST 构建器 (gemini_builder.py)

#### 2.1 代码执行工具注入逻辑
在 `prepare_gemini_rest_api_request()` 的 tools 构建部分：

```python
# 在 line 271-275 附近，googleSearch 注入后
if should_enable_code_execution(chat_input, request_id):
    gemini_tools_payload.append({"codeExecution": {}})
    logger.info(f"{log_prefix}: Enabled Code Execution tool for Gemini.")
```

#### 2.2 启用判断辅助函数
```python
def should_enable_code_execution(chat_input: ChatRequestModel, request_id: str) -> bool:
    """
    判断是否应启用代码执行工具
    - enable_code_execution=True: 强制开启
    - enable_code_execution=False: 强制关闭
    - enable_code_execution=None: 智能判断（检测用户意图）
    """
    log_prefix = f"RID-{request_id}"
    
    # 显式控制
    if chat_input.enable_code_execution is True:
        return True
    if chat_input.enable_code_execution is False:
        return False
    
    # Auto模式：基于模型版本和用户意图
    model_lower = chat_input.model.lower()
    
    # 仅 2.0/2.5 系列支持
    if not ("gemini-2.0" in model_lower or "gemini-2.5" in model_lower or "gemini-2-" in model_lower):
        return False
    
    # 检测用户意图关键词
    user_texts = extract_user_texts_from_parts_messages(chat_input.messages)
    intent_keywords = [
        "计算", "求解", "运行代码", "执行代码", "画图", "绘制", "plot", 
        "matplotlib", "数据分析", "统计", "csv", "pandas", "numpy",
        "calculate", "compute", "run code", "execute", "draw", "chart"
    ]
    
    user_text_lower = user_texts.lower()
    if any(keyword in user_text_lower for keyword in intent_keywords):
        logger.info(f"{log_prefix}: Auto-enabled code execution based on user intent")
        return True
    
    return False
```

### 3. OpenAI 兼容构建器 (openai_builder.py)

#### 3.1 代码执行工具注入
在 `prepare_openai_request()` 的 Gemini 特殊处理部分（line 138-149 附近）：

```python
# 在 google_search 注入后
if should_enable_code_execution_openai(request_data, request_id):
    google_tools = google_section.get("tools") or []
    if not any(tool.get("code_execution") is not None for tool in google_tools):
        google_tools.append({"code_execution": {}})
        google_section["tools"] = google_tools
        extra_body["google"] = google_section
        payload["extra_body"] = extra_body
        logger.info(f"RID-{request_id}: Enabled code_execution tool for Gemini (OpenAI format)")
```

### 4. Gemini 流解析 (gemini.py)

#### 4.1 解析 executableCode 和 codeExecutionResult
在 `stream_generator()` 的 SSE 解析循环中（line 398-446 附近）：

```python
for part in content_parts:
    # 现有的 thought/inlineData/text 处理...
    
    # 新增：代码执行片段
    elif "executableCode" in part:
        code_data = part.get("executableCode", {})
        code_text = code_data.get("code", "")
        code_lang = code_data.get("language", "python")
        if code_text:
            yield await sse_event_serializer_rest(AppStreamEventPy(
                type="code_executable",
                executable_code=code_text,
                code_language=code_lang
            ))
    
    # 新增：代码执行结果
    elif "codeExecutionResult" in part:
        result_data = part.get("codeExecutionResult", {})
        output_text = result_data.get("output", "")
        outcome = result_data.get("outcome", "success")
        if output_text or outcome == "error":
            yield await sse_event_serializer_rest(AppStreamEventPy(
                type="code_execution_result",
                code_execution_output=output_text,
                code_execution_outcome=outcome
            ))
```

### 5. OpenAI 兼容流解析 (openai.py)

OpenAI 兼容格式下，Gemini 的代码执行结果会在 delta.content 中以特殊格式返回，需要解析。

暂时不实现，因为 OpenAI 兼容格式的代码执行支持可能有限。优先实现 Gemini REST 原生支持。

### 6. 辅助函数 (helpers.py)

```python
def is_gemini_2_x_model(model_name: str) -> bool:
    """检测是否为 Gemini 2.x 或 2.5.x 系列模型"""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return ("gemini-2.0" in model_lower or 
            "gemini-2.5" in model_lower or 
            "gemini-2-" in model_lower)
```

## 测试用例

### 测试1：数学计算
```
用户: 前50个质数的和是多少？请生成并运行代码计算。
预期: 自动启用代码执行，返回代码片段和结果
```

### 测试2：数据可视化
```
用户: 用matplotlib画一个正弦波
预期: 自动启用代码执行，返回代码和图表（base64图片）
```

### 测试3：显式控制
```
请求: {"enableCodeExecution": false, ...}
预期: 即使问题涉及计算，也不启用代码执行
```

## 前端适配建议

### 事件订阅
```kotlin
when (event.type) {
    "code_executable" -> {
        // 显示代码片段（可折叠/复制）
        showCodeBlock(event.executableCode, event.codeLanguage)
    }
    "code_execution_result" -> {
        // 显示执行结果
        showExecutionResult(event.codeExecutionOutput, event.codeExecutionOutcome)
    }
}
```

### UI 建议
- 代码块：语法高亮、复制按钮、折叠/展开
- 执行结果：独立区域，支持图表内嵌显示
- 开关：设置中增加"代码执行"选项（auto/on/off）

## 安全与限制

1. **沙箱环境**: 代码在 Google 服务器的隔离环境中执行，30秒超时
2. **支持库**: 仅限官方文档列出的库（numpy, pandas, matplotlib等）
3. **无自定义安装**: 用户无法安装额外的 Python 包
4. **重试机制**: 代码错误时模型可自动重试（最多5次）

## 实施顺序

1. ✅ 数据模型扩展
2. ✅ Gemini REST 构建器改动
3. ✅ Gemini 流解析改动
4. ⏳ 测试与调试
5. ⏳ 文档更新
6. ⏳ 前端适配（可选）