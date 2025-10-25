# EzTalk Proxy 服务重构提案

目标：降低体量与耦合，提高可测试性，逐步无感迁移。

一、当前痛点与拆分原则
- 体量过大：单文件 800–900+ 行，职责混杂。
- 逻辑耦合：流式处理、修复策略、请求构建交叉调用。
- 测试困难：缺少可替换的窄接口与纯函数单元。
- 迁移风险：需要保持对外接口与日志的稳定。

拆分原则
- 单一职责：每个模块只做一件事。
- 组合优先：通过小组件组装成管线。
- 稳定门面：旧入口函数原样保留为适配层。
- 可测先行：对纯函数/策略提供独立单测。

二、目标目录结构

services/
├── streaming/                  # 流式处理流水线（拆自 stream_processor.py）
│   ├── __init__.py
│   ├── sse_models.py           # 轻量类型/常量、工具型纯函数
│   ├── content_state.py        # 状态与增量指针、去重
│   ├── reasoning.py            # 思考/推理片段抽取、<think> 标签
│   ├── delta.py                # 文本合并、重复检测、分块优化
│   ├── flush.py                # 刷新策略（检测块类型/安全刷新点）
│   ├── finalizer.py            # 尾部修复/供应商特殊格式化
│   ├── cleanup.py              # 错误/清理阶段收尾
│   └── processor.py            # 组合入口：process_openai_like_sse_stream(...)
├── requests/                   # 请求构建（拆自 request_builder.py）
│   ├── __init__.py
│   ├── system_prompt.py        # 意图/语言识别 + system 注入
│   ├── headers.py              # 鉴权头/供应商头组合
│   ├── converters.py           # parts ↔ REST 内容转换
│   ├── openai_builder.py       # prepare_openai_request(...)
│   ├── gemini_builder.py       # prepare_gemini_rest_api_request(...)
│   └── facade.py               # 统一门面，供旧代码调用
└── format_repair/              # 输出修复（拆自 format_repair.py）
    ├── __init__.py
    ├── core.py                 # FormatRepairService 组装器
    └── strategies/             # 策略实现（可独立单测）
        ├── __init__.py
        ├── code_strategy.py
        ├── math_strategy.py
        ├── json_strategy.py
        ├── table_strategy.py
        └── general_strategy.py

三、对外兼容与适配
- 保留原文件，改造为薄门面：
  - process_openai_like_sse_stream、handle_stream_error、handle_stream_cleanup 由 streaming/processor 调用实现。
  - prepare_openai_request、prepare_gemini_rest_api_request 由 requests/facade 导出。
  - format_repair_service、detect_output_type、batch_repair 由 format_repair/core 组合并导出。
- 日志类别与关键字段保持不变，避免可观测性破坏。
- 配置读取路径保持不变；新增策略开关通过原有 config 注入。

四、单元测试计划（pytest）
- streaming
  - content_state：去重、指针推进、边界条件
  - reasoning：<think> 抽取、供应商兼容
  - flush：刷新时机判定、代码围栏保护
  - finalizer：尾部修复兜底、智谱增量格式化守护
- requests
  - system_prompt：意图/语言检测样例
  - converters：parts ↔ REST 映射、角色映射
  - openai/gemini builder：payload/headers 组装与覆盖
- format_repair
  - 各策略：表格保护、代码围栏、JSON 修复、数学轻度修复

五、迁移步骤与验收
1) 引入新目录与空实现，落地接口签名并加测试桩。
2) 逐步把纯函数与策略移动到新模块，保证测试通过。
3) 将旧文件替换为薄门面，导入新实现，回归现有集成测试。
4) 对关键路径做回放比对：同一输入流 → 同一事件序列。
5) 增补极端样例（大段空白、未闭合 fenced、表格粘连）。

六、切面与非功能需求
- 日志：保留 RID 前缀与关键统计，新增模块化 logger 名称。
- 性能：保持 O(1) 增量发送，避免重复拷贝与多次修复。
- 安全：任何“修复导致空增量”启用回退原文的守护。

七、风险与回滚
- 行为差异风险：先由门面透传，再逐步切换具体实现。
- 回滚策略：门面层保留“旧实现”开关，出现问题可一键回退。

八、里程碑
- PR-1：目录与接口骨架 + 基础单测
- PR-2：streaming 拆分与门面回接
- PR-3：requests/format_repair 策略化与单测覆盖

附：保持兼容的旧入口
- streaming 门面：与旧 [stream_processor.py](backdAiTalk/eztalk_proxy/services/stream_processor.py) 同名函数签名不变。
- requests 门面：与旧 [request_builder.py](backdAiTalk/eztalk_proxy/services/request_builder.py) 入口一致。
- 修复服务：与旧 [format_repair.py](backdAiTalk/eztalk_proxy/services/format_repair.py) 的 format_repair_service 全局实例一致。

完毕。