# EzTalk Streaming NDJSON 契约（端到端）

本契约用于串联「上游 - 代理 - App 解析」完整链路的端到端校验。代理将应用内统一事件 `AppStreamEvent` 序列化为 NDJSON（每行一个 JSON 对象），客户端按行增量解析。

约定仅包含跨端稳定字段；新增字段必须向后兼容（客户端需 `ignoreUnknownKeys`）。

## 基础约定

- 传输介质：SSE（`text/event-stream`），每个 `data:` 行对应一个 NDJSON 事件对象。
- 回放介质：NDJSON 文件（每行一个事件 JSON），供本地/CI 复播。
- 时间戳：可选字段，UTC ISO8601 字符串；用于调试/排序，不参与语义断言。

## 事件类型与字段

所有事件包含字段：
- `type: string` 事件类型标识

以下为已对齐 Android 端的稳定子集（与 [AppStreamEvent](KunTalkwithAi/app1/app/src/main/java/com/example/everytalk/data/network/AppStreamEvent.kt) 一致）：

1) content
- 字段：
  - `text: string` 必填；累计全文或增量文本，当前策略为累计全文覆盖
  - `output_type: string | null` 可选；渲染建议：`general | code | table ...`
  - `block_type: string | null` 可选；块类型：`text | code_block | heading`
  - `timestamp: string | null` 可选
- 语义：内容中间态，UI可覆盖式更新（按 block_type）

2) content_final
- 字段同 `content`
- 语义：内容最终态（尾包修复后），必须在 `finish` 前出现（若有内容）

3) reasoning
- 字段：
  - `text: string` 必填
- 语义：思维链/推理内容，不与 content 混用

4) status_update
- 字段：
  - `stage: string` 必填
- 语义：进度提示，UI 可选择显示为状态行

5) web_search_results
- 字段：
  - `results: Array<{title: string, href: string, snippet: string}>`
- 语义：联网搜索结果列表

6) error
- 字段：
  - `message: string` 必填
  - `upstreamStatus: number | null` 可选
- 语义：友好错误信息；错误后仍应有 `finish` 终结

7) finish
- 字段：
  - `reason: string` 必填；`stop | stream_end | error_in_stream | upstream_error_or_connection_failed ...`
- 语义：会话结束标志。任意场景都必须有且仅有一次。

注：后端内部存在 `reasoning_finish` 等辅助事件，但为避免客户端反序列化失败，不纳入公共契约与回放快照。

## 示例（NDJSON 片段）

```json
{"type":"reasoning","text":"Thinking A..."}
{"type":"content","text":"Hello","output_type":"general","block_type":"text","timestamp":"2025-01-01T00:00:00Z"}
{"type":"content","text":"Hello, wor","output_type":"general","block_type":"text","timestamp":"2025-01-01T00:00:01Z"}
{"type":"content","text":"Hello, world!\n\nThis is a list:\n1. A\n2. B\n","output_type":"general","block_type":"text","timestamp":"2025-01-01T00:00:02Z"}
{"type":"content_final","text":"Hello, world!\n\nThis is a list:\n1. A\n2. B\n","output_type":"general","block_type":"text","timestamp":"2025-01-01T00:00:03Z"}
{"type":"finish","reason":"stop"}
```

## 端到端校验流程

- 生成快照（后端本地脚本/测试）
  - 从上游返回的 OpenAI-like SSE json 逐行进入处理器 [process_openai_like_sse_stream()](backdAiTalk/eztalk_proxy/services/stream_processor.py:294)，捕获协议事件对象，序列化为 NDJSON 行
  - 将 NDJSON 写入 `snapshots/` 目录（见下方约定）
- Android 回放（单测）
  - 从 `src/test/resources/stream_contract/*.ndjson` 按行读取
  - 使用 `kotlinx.serialization.Json(ignoreUnknownKeys=true)` 反序列化为 [AppStreamEvent](KunTalkwithAi/app1/app/src/main/java/com/example/everytalk/data/network/AppStreamEvent.kt)
  - 逐条喂入 [MessageProcessor.processStreamEvent()](KunTalkwithAi/app1/app/src/main/java/com/example/everytalk/util/messageprocessor/MessageProcessor.kt) ，断言 UI 最终文本与块结构

## 目录约定

- 后端快照脚本输出（建议，不强制）：
  - `backdAiTalk/tests/fixtures/stream_contract/*.ndjson`
- Android 测试用回放快照：
  - `KunTalkwithAi/app1/app/src/test/resources/stream_contract/*.ndjson`

## 断言建议

- 顺序保障：所有事件按时间线顺序；`finish` 必为最后一行
- 内容幂等：对同一 content 覆盖更新不产生重复
- 结构一致：`block_type` 变化应触发块覆盖或切换，不追加重复块
- 错误路径：出现 `error` 时，仍最终产生 `finish`

## 变更治理

- 新增事件类型：先在客户端增加可选分支（未知事件忽略），再在后端启用
- 字段新增：仅新增可选字段；严禁删除/重命名已发布字段
- 版本化：必要时引入 `meta.version` 帧（独立事件），或快照文件命名带版本后缀
