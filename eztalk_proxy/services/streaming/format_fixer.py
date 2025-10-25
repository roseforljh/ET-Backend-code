"""
Markdown格式修复模块
专门处理AI输出的不规范Markdown格式
"""
import re
import logging

logger = logging.getLogger("EzTalkProxy.FormatFixer")


class MarkdownFormatFixer:
    """修复AI输出的不规范Markdown格式"""
    
    PATTERNS = {
        # 标题格式：确保 # 后有空格
        'heading_space': re.compile(r'^(#{1,6})([^\s#\n])', re.MULTILINE),
        # 标题末尾多余的 # 符号（移除所有末尾的#，包括###这种）
        'heading_trailing_hash': re.compile(r'^(#{1,6}\s+[^#\n]+?)(#{1,})$', re.MULTILINE),
        # 行中出现的标题标记（非行首的##）
        'inline_heading': re.compile(r'([^\n#])(#{2,6})\s*([^\n]+)'),
        # 列表格式：确保 - 后有空格
        'list_space': re.compile(r'^-([^\s\-\n])', re.MULTILINE),
        # 列表前缺少换行
        'list_newline': re.compile(r'([^\n\-\*\+])([-*+]\s)'),
        # 多余的标记符号（超过3个#）
        'excess_hash': re.compile(r'#{4,}'),
        # 分隔线前的多余内容（如 $2.45---）
        'separator_cleanup': re.compile(r'([^\n\s])(---+)'),
    }
    
    @classmethod
    def fix(cls, text: str, aggressive: bool = False) -> str:
        if not text or not text.strip():
            return text
        if not aggressive:
            if '```' in text:
                logger.debug("Skipping format fix: contains code fence")
                return text
        fixed = text
        try:
            # 1) 行内标题 → 独立行
            fixed = cls.PATTERNS['inline_heading'].sub(r'\1\n\2 \3', fixed)
            # 2) 标题 # 后补空格
            fixed = cls.PATTERNS['heading_space'].sub(r'\1 \2', fixed)
            # 3) 过多 # 收敛到 ###（先于移除尾部#）
            fixed = cls.PATTERNS['excess_hash'].sub('###', fixed)
            # 4) 去掉标题尾部多余 #
            fixed = cls.PATTERNS['heading_trailing_hash'].sub(r'\1', fixed)

            # 5) 表格保护：对包含表格管道的行，不应用会影响连字符的规则
            def _process_line(line: str) -> str:
                if '|' in line:
                    # 视为可能的表格行，避免列表/分隔线规则误伤
                    return line
                # 列表短横后补空格
                line = cls.PATTERNS['list_space'].sub(r'- \1', line)
                # 分隔线（---）前补换行，但跳过表格（已在前面 return）
                line = cls.PATTERNS['separator_cleanup'].sub(r'\1\n\2', line)
                return line

            lines = fixed.split('\n')
            lines = [_process_line(ln) for ln in lines]

            # 5.1) 表格对齐分隔行重建：修复被拆成“|:|”“--”等碎片的情况
            def _is_table_header(s: str) -> bool:
                return s.strip().startswith('|') and s.strip().endswith('|') and s.count('|') >= 2

            def _is_alignment_piece(s: str) -> bool:
                t = s.strip()
                if not t:
                    return False
                # 常见碎片：|:|、| : |、--、---、:---、---:、| :--- | 等
                if t in ('--', '---', '|:|', '| : |'):
                    return True
                # 仅含管道与冒号或短横
                if set(t) <= set('|:- '):
                    # 但过滤纯管道
                    return any(ch in t for ch in ':-')
                return False

            i = 0
            rebuilt: list[str] = []
            while i < len(lines):
                line = lines[i]
                if _is_table_header(line):
                    # 统计列数
                    cells = [c for c in line.strip().split('|')[1:-1]]
                    col_count = max(1, len(cells))
                    j = i + 1
                    piece_lines: list[str] = []
                    while j < len(lines) and _is_alignment_piece(lines[j]):
                        piece_lines.append(lines[j])
                        j += 1
                    if piece_lines:
                        # 构建标准左对齐分隔行
                        align_cell = ' :--- '
                        rebuilt.append(line)
                        rebuilt.append('|' + ('{}|'.format(align_cell) * col_count))
                        # 跳过碎片行
                        i = j
                        continue
                rebuilt.append(line)
                i += 1

            fixed = '\n'.join(rebuilt)

            # 6) 去掉“表头行与分隔行”之间的多余空行，便于渲染器识别为表格
            # 形如：
            # | a | b |
            #
            # | --- | --- |
            fixed = re.sub(
                r'(^\|.*\|\s*)\n\s*\n(\s*\|[ :\-|:]+)',
                r'\1\n\2',
                fixed,
                flags=re.MULTILINE
            )
            # 注意：不再做任何 ** 加粗 的修复，避免破坏语法
            
            if fixed != text:
                logger.debug(f"Format fixed: {len(text)} -> {len(fixed)} chars")
        except Exception as e:
            logger.error(f"Error fixing markdown format: {e}")
            return text
        return fixed
    
    @classmethod
    def fix_streaming_chunk(cls, chunk: str) -> str:
        if not chunk or len(chunk) < 3:
            return chunk
        fixed = chunk
        try:
            fixed = cls.PATTERNS['heading_space'].sub(r'\1 \2', fixed)
            fixed = cls.PATTERNS['list_space'].sub(r'- \1', fixed)
        except Exception as e:
            logger.error(f"Error fixing streaming chunk: {e}")
            return chunk
        return fixed


def fix_markdown_format(text: str, aggressive: bool = False) -> str:
    return MarkdownFormatFixer.fix(text, aggressive=aggressive)