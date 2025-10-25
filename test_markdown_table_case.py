#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from eztalk_proxy.services.streaming.format_fixer import fix_markdown_format

text = """好的，我们来解释一下“国民度”这个词。
### # 与相关词语的区别
为了更好地理解，我们可以将它与另外两个词进行比较：

| 特征 | **国民度** | **流量** | **知名度** |
|:|

--

|:|

--
| - |
| **核心** | 大众认可与喜爱 | 粉丝热度与数据 | 被人知道的程度 |
| **人群** | 全体国民，跨年龄层 | 特定粉丝群体，偏年轻化 | 不限，但可能局限于特定领域 |
| **评价** | **普遍为正面** | 中性，依赖粉丝维护 | 中性或负面（如因丑闻出名） |
| **例子** | 演员刘德华、科学家袁隆平、电视剧《西游记》 | 某些数据很高但大众不熟的偶像明星 | 因某个社会事件而出名的人 |"""

fixed = fix_markdown_format(text, aggressive=True)
print("=== FIXED ===")
print(fixed)