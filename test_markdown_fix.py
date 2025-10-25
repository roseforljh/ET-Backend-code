#!/usr/bin/env python3
"""
测试Markdown格式修复功能
"""
from eztalk_proxy.services.streaming.format_fixer import fix_markdown_format

# 测试用例：你提供的实际文本
test_text = """AI在真实市场交易## N01推出的Alpha Arena平台###

实况 | 排行榜 | 模型- BTC $110,801.50- ETH $4,033.65- SOL $192.30- BNB $1,112.55- DOGE $0.2004- XRP $2.45---

### 模型对话（DEEPSEEK CHAT V3.1）

**10/2020:53:38**我目前持有ETH、SOL、XRP、BTC、DOGE和BNB的所有仓位，因为它们的**失效条件均未触发**"""

print("=" * 60)
print("原始文本:")
print("=" * 60)
print(test_text)
print("\n" + "=" * 60)
print("修复后:")
print("=" * 60)
fixed = fix_markdown_format(test_text, aggressive=False)
print(fixed)
print("\n" + "=" * 60)
print("修复统计:")
print("=" * 60)
print(f"原始长度: {len(test_text)}")
print(f"修复后长度: {len(fixed)}")
print(f"是否有变化: {'是' if fixed != test_text else '否'}")