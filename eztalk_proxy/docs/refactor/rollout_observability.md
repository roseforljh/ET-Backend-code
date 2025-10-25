# 格式修复灰度与可观测说明

本文档描述格式修复（Format Repair）的灰度/影子执行、可观测与回滚策略，适用于 EzTalk Proxy 流式输出路径。

## 配置项（全局）

配置文件：backdAiTalk/eztalk_proxy/config/format_repair_config.json

关键字段：

- rollout_mode: on | off | percentage | shadow
- rollout_percentage: 0..100 （percentage 模式有效）
- log_sampling_rate: 0.0..1.0（日志采样）
- shadow_compare_preview_chars: 影子对比日志中预览字符数
- enable_format_repair: 全局 Kill Switch

说明：

- on: 全量开启，保持历史行为
- off: 全量关闭（快速回滚）
- percentage: 按请求维度稳定哈希进桶做百分比灰度
- shadow: 影子执行，实际输出保持“原生 AI 文本”，仅记录对比日志

默认值：

- rollout_mode: on
- rollout_percentage: 100
- log_sampling_rate: 0.1
- shadow_compare_preview_chars: 120

## 请求级覆盖（前端可透传）

支持在请求体中通过 customModelParameters.formatRepair 进行覆盖：

{
  "customModelParameters": {
    "formatRepair": {
      "rolloutModeOverride": "shadow",
      "rolloutPercentageOverride": 20,
      "logSamplingOverride": 1.0,
      "disableFormatRepair": false
    }
  }
}

字段语义：

- rolloutModeOverride: off | on | percentage | shadow（优先于全局）
- rolloutPercentageOverride: 0..100
- logSamplingOverride: 0.0..1.0
- disableFormatRepair: true 时强制跳过修复

## 接入位置（调用点）

- 预处理：preprocess_ai_output_content（通用修复）
- 实时增量：CONTENT_FLUSH 路径（幂等轻量修复）
- 最终尾块：finalize_delta（尾部安全修复与类型判定）

三处均调用统一判定助手 decide_repair_action：

- action=apply: 执行修复
- action=skip: 跳过修复，直接透传
- action=shadow: 影子执行，结果仅用于日志对比

判定包含原因码 reason_code，便于回溯（如 mode_on / mode_off / pct_apply_30 / pct_skip_30 / mode_shadow）。

## 可观测与日志

日志前缀包含请求 ID：RID-xxx

新增关键日志（按采样）：

- ROLLOUT_PREPROCESS: 预处理阶段决策（apply）
- ROLLOUT_PREPROCESS_SHADOW_DIFF: 预处理影子结果差异预览
- ROLLOUT_REALTIME: 实时增量阶段决策（apply）
- ROLLOUT_REALTIME_SHADOW_DIFF: 实时影子差异预览
- ROLLOUT_FINAL: 最终修复阶段决策（apply）
- ROLLOUT_FINAL_SHADOW_DIFF: 最终影子差异预览

预览长度由 shadow_compare_preview_chars 控制，默认 120。

采样率由 log_sampling_rate 控制，默认 0.1。采样基于请求 ID 与阶段名的稳定哈希，避免日志风暴。

## 回滚策略（Kill Switch）

- 全量关闭：enable_format_repair=false
- 强制跳过：请求级 disableFormatRepair=true
- 快速降级：rollout_mode=off 或 rollout_mode=percentage + rollout_percentage=0

所有分支均保留“若修复后变空则回退到原文”的保护逻辑。

## 验证与测试

单元测试：backdAiTalk/tests/test_rollout_and_shadow.py

覆盖点：

- 模式选择：on/off/percentage/shadow
- 百分比边界：0% 跳过；100% 全量应用
- 请求覆盖：customModelParameters.formatRepair.* 生效
- 禁用优先级：disableFormatRepair=true 强制跳过

建议补充集成测试验证：

- 影子执行不影响前端展示
- apply/skip 在三处调用点行为一致

## 常见操作示例

- 临时影子观测（全量）：rollout_mode=shadow, log_sampling_rate=1.0
- 小流量灰度：rollout_mode=percentage, rollout_percentage=5
- 紧急关停：enable_format_repair=false

## 风险与注意

- 供应商特定格式化（如智谱）仍受 prevent_empty_delta_after_formatting 保护
- 表格/代码围栏下最终修复默认跳过，避免破坏结构
- 影子日志仅采样输出，避免敏感/长文本泄露到日志

以上方案在保持历史行为兼容的同时，提供灰度与可观测能力，支持快速回滚与精细化排查。