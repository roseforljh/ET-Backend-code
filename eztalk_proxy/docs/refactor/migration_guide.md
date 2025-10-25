# EzTalk Proxy 重构迁移指南

本指南说明从单体服务迁移到模块化流水线/策略化实现的步骤、开关、回滚方案与测试方法。目标是在保持对外行为兼容的前提下，逐步完成可维护性和可测试性提升。

## 1. 模块边界与目录

- 流式处理（拆自 services/stream_processor.py）
  - services/streaming/
    - processor.py（门面，组合入口）
    - reasoning.py（思考标签抽取）
    - content_state.py（哈希/去重）
    - delta.py（增量合并）
    - flush.py（代码围栏/块检测/安全刷新判断）
    - finalizer.py（尾部修复 + 供应商格式化 + 类型判定）
    - cleanup.py（清理阶段尾部刷新与事件构造）
- 请求构建（拆自 services/request_builder.py）
  - services/requests/
    - facade.py（门面）
    - system_prompt.py（意图/语言 + 系统提示注入，当前委托旧实现）
    - headers.py（鉴权与供应商特定头部）
    - converters.py（消息转换，当前委托旧实现）
- 格式修复（拆自 services/format_repair.py）
  - services/formatting/
    - core.py（门面/组合器，保留旧实现回退）
    - strategies/（策略集合：code/json/math/table/general，当前委托旧实现）

## 2. 行为兼容与切换开关

- 默认行为：保持与旧实现完全一致
  - services/stream_processor.py 对外接口未变，内部逐步委托到 streaming 子模块
  - services/requests/facade.py 默认仍使用旧的 payload/url 逻辑，但已将 headers/system_prompt 等分离（并在不改变行为的前提下合并头部）
  - services/formatting/core.py 默认返回旧的 `format_repair_service` 实例

- 策略引擎开关（格式修复）
  - 配置文件：services/format_config.py 中添加 `enable_strategy_engine: bool = False`
  - 设为 True 时，core.py 将返回策略组合服务（仍调用旧的 detect_output_type，以避免误判）
  - 策略文件位于 services/formatting/strategies/

- 供应商/格式化保护开关（原有）
  - 防空增量参数：prevent_empty_delta_after_(realtime|final)_repair / prevent_empty_delta_after_formatting
  - 供应商特定格式化开关：enable_provider_specific_formatting

## 3. 渐进迁移建议

1) 流水线继续拆分，完成 processor 组装  
   - 将剩余解析/聚合/工具调用等小逻辑迁出到 streaming 下独立模块（保留门面委托），持续回归测试

2) 请求构建策略化  
   - 使用 services/requests/system_prompt.py/headers.py/converters.py 持续替换 facade 内逻辑；保留 legacy 回退  
   - 增补细粒度测试覆盖（OpenAI 兼容 + Gemini REST，含思考配置）

3) 格式修复策略化  
   - 按需扩展 strategies/ 下的策略；启用 `enable_strategy_engine=True` 后进行 A/B 回归对比  
   - 若发现差异，保持开关默认关闭，逐步迭代策略

## 4. 回滚策略

- 流式处理：processor 门面仍可直接使用旧实现（当前已委托旧实现路径可随时回退）
- 请求构建：facade 对 legacy 的调用仍保留；headers/system_prompt 等分离为“附加层”，可关闭或回退合并逻辑
- 格式修复：将 `enable_strategy_engine` 设为 False 即回退到旧服务实例

## 5. 测试与验证

- 运行测试（建议使用虚拟环境）  
  ```bash
  python -m pytest -q backdAiTalk/tests
  ```
- 当前测试覆盖（22+ 用例）  
  - 流式辅助与刷新/尾部修复边界  
  - 增量合并/去重  
  - 请求构建：意图/语言/系统提示注入  
  - 格式修复：表格粘连/未闭合代码块/JSON 修复/类型判定  
- 若启用策略引擎：  
  - 在 format_config.py 启用 `enable_strategy_engine = True` 后再次运行测试，并对比关键输出路径日志与增量事件序列

## 6. 日志与可观测性

- 保持原日志类别与 RID 前缀，保留关键统计（total/content/reasoning chunks 等）
- 新模块增加子 logger 名称（如 EzTalkProxy.StreamProcessors.Finalizer/Cleanup），便于精准定位问题

## 7. 常见问题

- Windows 上 pytest 启动器警告  
  - 使用 `python -m pytest -q backdAiTalk/tests` 可规避  
- 导入路径问题  
  - 测试中已添加 repo 根路径到 sys.path。若需单独运行，确保从仓库根目录执行测试命令。

## 8. 里程碑与完成状态

- 已完成：  
  - 门面/骨架搭建、首批拆分（reasoning、content_state、delta、flush、finalizer、cleanup）、请求 headers 分离、回归测试  
- 待完成：  
  - 请求构建策略化落地（system_prompt/converters），并在 facade 中切换  
  - 格式修复策略引擎正式启用并经 A/B 验证后默认开启  
  - README/文档持续补充最终切换步骤与开关说明

如需强制回退到旧行为，仅需：  
- 在 format_config.py 中保持 `enable_strategy_engine = False`  
- 在 streaming/processor.py 门面中改回直接委托旧实现（当前已保留委托路径）  
- requests/facade.py 保持 legacy 计算 URL/payload（当前保留）