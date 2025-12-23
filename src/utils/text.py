import re
from typing import List

def truncate_for_log(text: any, limit: int = 400) -> str:
    """Safe truncation for logging."""
    try:
        s = str(text)
    except Exception:
        return "<unprintable>"
    return s if len(s) <= limit else s[:limit] + "...[truncated]"

def chunk_text(text: str, chunk_level: int, is_pdf_content: bool = False) -> List[str]:
    """Split text into chunks based on the configured chunk level"""
    # If no chunking requested, return the whole text as a single chunk
    if chunk_level <= 0:
        return [text]

    # Level 1: Phrase-level chunking (split by commas, colons, semicolons)
    if chunk_level == 1:
        paragraphs = text.split("\n")
        chunks = []
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            paragraph_phrases = re.split(r"(?<=[,;:])\s+", paragraph)
            for phrase in paragraph_phrases:
                if phrase.strip():
                    chunks.append(phrase.strip())
        return chunks

    # Level 2: Sentence-level chunking
    if chunk_level == 2:
        if is_pdf_content:
            chunks = []
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                if sentence.strip():
                    chunks.append(sentence.strip())
        else:
            paragraphs = text.split("\n")
            chunks = []
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                for sentence in sentences:
                    if sentence.strip():
                        chunks.append(sentence.strip())
        return chunks

    # Level 3: Paragraph-level chunking
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if chunk_level == 3:
        return paragraphs

    # Level 4-10: Multi-paragraph chunking
    chunks = []
    paragraphs_per_chunk = chunk_level - 2
    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        chunk = "\n".join(paragraphs[i : i + paragraphs_per_chunk])
        chunks.append(chunk)
    return chunks

async def clean_text_formatting(content: str) -> str:
    """Clean text formatting by merging short lines and handling repeated character patterns"""
    from src.constants import NUMBERED_LINE_PATTERNS
    
    lines = content.split("\n")
    cleaned_lines = []

    for line in lines:
        # Repeated characters (5+ identical characters)
        repeated_char_pattern = re.compile(r"((.)\2{4,})")
        matches = list(repeated_char_pattern.finditer(line))
        if matches:
            for match in reversed(matches):
                char_sequence = match.group(1)
                char = match.group(2)
                if len(char_sequence) >= 5:
                    replacement = char * 2 + "(...)" + char * 2
                    start, end = match.span()
                    line = line[:start] + replacement + line[end:]

        # Repeated patterns (abc abc abc abc abc)
        for pattern_length in range(2, 4):
            i = 0
            while i <= len(line) - pattern_length * 5:
                pattern = line[i : i + pattern_length]
                repetition_count = 0
                for j in range(i, len(line) - pattern_length + 1, pattern_length):
                    if line[j : j + pattern_length] == pattern:
                        repetition_count += 1
                    else:
                        break
                if repetition_count >= 5:
                    replacement = pattern * 2 + "(...)" + pattern * 2
                    total_length = pattern_length * repetition_count
                    line = line[:i] + replacement + line[i + total_length :]
                i += 1

        # Repeated ellipsis patterns
        ellipsis_pattern = re.compile(r"(\S\S\(\.\.\.\)\S\S\s+)(\1){2,}")
        ellipsis_matches = list(ellipsis_pattern.finditer(line))
        if ellipsis_matches:
            for match in reversed(ellipsis_matches):
                single_instance = match.group(1)
                start, end = match.span()
                line = line[:start] + single_instance + line[end:]
        cleaned_lines.append(line)

    # Short line merging
    lines = cleaned_lines
    merged_lines = []
    short_line_group = []
    mixed_case_pattern = re.compile(r"[a-z][A-Z]")

    i = 0
    while i < len(lines):
        current_line = lines[i].strip()
        word_count = len(current_line.split())
        if word_count <= 5 and current_line:
            is_numbered_item = False
            for pattern in NUMBERED_LINE_PATTERNS:
                if re.search(pattern, current_line):
                    is_numbered_item = True
                    break
            if is_numbered_item and short_line_group:
                prev_line = short_line_group[-1]
                prev_match = re.search(r"(\d+)[\.\)\:]", prev_line)
                curr_match = re.search(r"(\d+)[\.\)\:]", current_line)
                if prev_match and curr_match:
                    try:
                        if int(curr_match.group(1)) != int(prev_match.group(1)) + 1:
                            is_numbered_item = False
                    except ValueError:
                        pass
            if is_numbered_item:
                if short_line_group:
                    for sl in short_line_group: merged_lines.append(sl)
                    short_line_group = []
                merged_lines.append(current_line)
            else:
                short_line_group.append(current_line)
        else:
            if short_line_group:
                if len(short_line_group) >= 5:
                    total_lc_to_uc = sum(1 for line in short_line_group for j in range(1, len(line)) if line[j-1].islower() and line[j].isupper())
                    has_mixed_case = (sum(1 for line in short_line_group if mixed_case_pattern.search(line)) >= len(short_line_group) * 0.3) or (total_lc_to_uc >= 3)
                    if merged_lines:
                        for j in range(min(2, len(short_line_group))): merged_lines[-1] += f". {short_line_group[j]}"
                        merged_lines.append("(Navigation menu removed)" if has_mixed_case else "(Headers removed)")
                        if len(short_line_group) >= 4:
                            merged_lines.append(short_line_group[-2])
                            merged_lines.append(short_line_group[-1])
                    else:
                        for j in range(min(2, len(short_line_group))): merged_lines.append(short_line_group[j])
                        merged_lines.append("(Navigation menu removed)" if has_mixed_case else "(Headers removed)")
                        if len(short_line_group) >= 4:
                            merged_lines.append(short_line_group[-2])
                            merged_lines.append(short_line_group[-1])
                else:
                    for j, sl in enumerate(short_line_group):
                        if j == 0 and merged_lines: merged_lines[-1] += f". {sl}"
                        else: merged_lines.append(sl)
                short_line_group = []
            if current_line: merged_lines.append(current_line)
        i += 1
    # Cleanup trailing short group
    if short_line_group:
        if len(short_line_group) >= 5:
            merged_lines.append("(Cleanup short group removed)")
        else:
            for sl in short_line_group: merged_lines.append(sl)
    return "\n".join(merged_lines)
