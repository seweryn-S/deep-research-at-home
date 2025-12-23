from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeTextMixin:
    async def count_tokens(self, text: str) -> int:
        """Approximate token count for text.

        Uses a lightweight heuristic based on whitespace-separated words to
        avoid relying on any specific tokenizer API.
        """
        if not text:
            return 0

        words = text.split()
        return int(len(words) * 1.3)  # Approximate token count using words

    async def extract_token_window(
        self, content: str, start_token: int, window_size: int
    ) -> str:
        """Extract a window of tokens from content"""
        try:
            # Get a rough estimate of tokens per character in this content
            total_tokens = await self.count_tokens(content)
            chars_per_token = len(content) / max(1, total_tokens)

            # Approximate character positions
            start_char = int(start_token * chars_per_token)
            window_chars = int(window_size * chars_per_token)

            # Ensure we don't go out of bounds
            start_char = max(0, min(start_char, len(content) - 1))
            end_char = min(len(content), start_char + window_chars)

            # Extract the window
            window_content = content[start_char:end_char]

            # Ensure we have complete sentences
            # Find the first sentence boundary
            if start_char > 0:
                first_period = window_content.find(". ")
                if first_period > 0 and first_period < len(window_content) // 10:
                    window_content = window_content[first_period + 2 :]

            # Find the last sentence boundary
            last_period = window_content.rfind(". ")
            if last_period > 0 and last_period > len(window_content) * 0.9:
                window_content = window_content[: last_period + 1]

            return window_content

        except Exception as e:
            logger.error(f"Error extracting token window: {e}")
            # If error, return a portion of the content
            if len(content) > 0:
                # Calculate safe window
                safe_start = min(
                    len(content) - 1,
                    max(0, int(len(content) * (start_token / total_tokens))),
                )
                safe_end = min(len(content), safe_start + window_size)
                return content[safe_start:safe_end]
            return content

    async def clean_text_formatting(self, content: str) -> str:
        """Clean text formatting by merging short lines and handling repeated character patterns"""
        # Handle repeated character patterns first
        # Split into lines to process each line individually
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            # Check for repeated characters (5+ identical characters in a row)
            repeated_char_pattern = re.compile(
                r"((.)\2{4,})"
            )  # Same character 5+ times
            matches = list(repeated_char_pattern.finditer(line))

            if matches:
                # Process each match in reverse order to avoid index shifts
                for match in reversed(matches):
                    char_sequence = match.group(1)
                    char = match.group(2)
                    if len(char_sequence) >= 5:
                        # Keep first 2 and last 2 instances, replace middle with (...)
                        replacement = char * 2 + "(...)" + char * 2
                        start, end = match.span()
                        line = line[:start] + replacement + line[end:]

            # Check for repeated character patterns (like abc abc abc abc)
            # Look for patterns of 2-3 chars that repeat at least 3 times
            for pattern_length in range(2, 4):  # Check for 2-3 character patterns
                i = 0
                while (
                    i <= len(line) - pattern_length * 5
                ):  # Need at least 5 repetitions
                    pattern = line[i : i + pattern_length]

                    # Check if this is a repeating pattern
                    repetition_count = 0
                    for j in range(i, len(line) - pattern_length + 1, pattern_length):
                        if line[j : j + pattern_length] == pattern:
                            repetition_count += 1
                        else:
                            break

                    # If we found a repeated pattern
                    if repetition_count >= 5:
                        # Keep first 2 and last 2 repetitions, replace middle with (...)
                        replacement = pattern * 2 + "(...)" + pattern * 2
                        total_length = pattern_length * repetition_count
                        line = line[:i] + replacement + line[i + total_length :]

                    i += 1

            # Check for repeated patterns with ellipsis that are created by earlier processing
            ellipsis_pattern = re.compile(r"(\S\S\(\.\.\.\)\S\S\s+)(\1){2,}")
            ellipsis_matches = list(ellipsis_pattern.finditer(line))

            if ellipsis_matches:
                # Process each match in reverse order to avoid index shifts
                for match in reversed(ellipsis_matches):
                    # Replace multiple repetitions with just one instance
                    single_instance = match.group(1)
                    start, end = match.span()
                    line = line[:start] + single_instance + line[end:]

            cleaned_lines.append(line)

        # Now handle short lines processing with semantic awareness
        lines = cleaned_lines
        merged_lines = []
        short_line_group = []

        # Define better mixed case pattern (lowercase followed by uppercase in the same word)
        # This will match patterns like: "PsychologyFor", "MediaPsychology", etc.
        mixed_case_pattern = re.compile(r"[a-z][A-Z]")

        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            word_count = len(current_line.split())

            # Check if this is a short line (5 words or fewer)
            if word_count <= 5 and current_line:
                # Check if it's part of a numbered list
                is_numbered_item = False

                # Match various numbering patterns:
                # - "1. Item"
                # - "1) Item"
                # - "1: Item"
                # - "A. Item"
                # - "A) Item"
                # - "A: Item"
                # - "Item 1."
                # - "Item 1)"
                # - "Item 1:"
                # Check if line matches any numbered pattern
                for pattern in NUMBERED_LINE_PATTERNS:
                    if re.search(pattern, current_line):
                        is_numbered_item = True
                        break

                # Check if this is part of a sequence of numbered items
                if is_numbered_item and short_line_group:
                    # Look for sequential numbering
                    prev_number = None
                    curr_number = None

                    # Try to extract numbers from current and previous line
                    prev_line = short_line_group[-1]
                    prev_match = re.search(r"(\d+)[\.\)\:]", prev_line)
                    curr_match = re.search(r"(\d+)[\.\)\:]", current_line)

                    if prev_match and curr_match:
                        try:
                            prev_number = int(prev_match.group(1))
                            curr_number = int(curr_match.group(1))

                            # Check if sequential
                            if curr_number == prev_number + 1:
                                is_numbered_item = True
                            else:
                                is_numbered_item = False
                        except ValueError:
                            pass

                # If it's a numbered item in a sequence, treat as normal text
                if is_numbered_item:
                    # Add it as separate line
                    if short_line_group:
                        # Flush any existing short lines
                        for j, short_line in enumerate(short_line_group):
                            merged_lines.append(short_line)
                        short_line_group = []

                    # Add this numbered item
                    merged_lines.append(current_line)
                else:
                    # Add to current group of short lines
                    short_line_group.append(current_line)
            else:
                # Process any existing short line group before adding this line
                if short_line_group:
                    # Check if we have 5 or more short lines in a sequence
                    if len(short_line_group) >= 5:
                        # Count mixed case occurrences in the group
                        mixed_case_count = 0
                        total_lc_to_uc = 0

                        for line in short_line_group:
                            # Count individual lowercase-to-uppercase transitions
                            for j in range(1, len(line)):
                                if (
                                    j > 0
                                    and line[j - 1].islower()
                                    and line[j].isupper()
                                ):
                                    total_lc_to_uc += 1

                            # Also check if the line itself has the mixed case pattern
                            if mixed_case_pattern.search(line):
                                mixed_case_count += 1

                        # If many lines have mixed case patterns or there are many transitions,
                        # they're likely navigation/menu items
                        has_mixed_case = (
                            mixed_case_count >= len(short_line_group) * 0.3
                        ) or (total_lc_to_uc >= 3)

                        # Keep first two and last two, replace middle with note
                        if merged_lines:
                            # Combine first two with previous line if possible
                            for j in range(min(2, len(short_line_group))):
                                merged_lines[-1] += f". {short_line_group[j]}"

                            # Add note about removed headers
                            if has_mixed_case:
                                merged_lines.append("(Navigation menu removed)")
                            else:
                                merged_lines.append("(Headers removed)")

                            # Add last two as separate lines
                            last_idx = len(short_line_group) - 2
                            if (
                                last_idx >= 2
                            ):  # Ensure we have lines left after removing middle
                                merged_lines.append(short_line_group[last_idx])
                                merged_lines.append(short_line_group[last_idx + 1])
                        else:
                            # If no previous line, handle differently
                            for j in range(min(2, len(short_line_group))):
                                merged_lines.append(short_line_group[j])

                            # Add note about removed headers or menu
                            if has_mixed_case:
                                merged_lines.append("(Navigation menu removed)")
                            else:
                                merged_lines.append("(Headers removed)")

                            last_idx = len(short_line_group) - 2
                            if last_idx >= 2:
                                merged_lines.append(short_line_group[last_idx])
                                merged_lines.append(short_line_group[last_idx + 1])
                    else:
                        # For small groups, merge with previous line if possible
                        for j, short_line in enumerate(short_line_group):
                            if j == 0 and merged_lines:
                                # First short line gets merged with previous
                                merged_lines[-1] += f". {short_line}"
                            else:
                                # Subsequent lines added separately
                                merged_lines.append(short_line)

                    # Reset short line group
                    short_line_group = []

                # Add current non-short line
                if current_line:
                    merged_lines.append(current_line)

            i += 1

        # Handle any remaining short line group
        if short_line_group:
            if len(short_line_group) >= 5:
                # Count mixed case occurrences in the group
                mixed_case_count = 0
                total_lc_to_uc = 0

                for line in short_line_group:
                    # Count individual lowercase-to-uppercase transitions
                    for j in range(1, len(line)):
                        if j > 0 and line[j - 1].islower() and line[j].isupper():
                            total_lc_to_uc += 1

                    # Also check if the line itself has the mixed case pattern
                    if mixed_case_pattern.search(line):
                        mixed_case_count += 1

                # If many lines have mixed case patterns or there are many transitions,
                # they're likely navigation/menu items
                has_mixed_case = (mixed_case_count >= len(short_line_group) * 0.3) or (
                    total_lc_to_uc >= 3
                )

                # Keep first two and last two, replace middle with note
                if merged_lines:
                    # Combine first two with previous line if possible
                    for j in range(min(2, len(short_line_group))):
                        merged_lines[-1] += f". {short_line_group[j]}"

                    # Add note about removed headers
                    if has_mixed_case:
                        merged_lines.append("(Navigation menu removed)")
                    else:
                        merged_lines.append("(Headers removed)")

                    # Add last two as separate lines
                    last_idx = len(short_line_group) - 2
                    if last_idx >= 2:
                        merged_lines.append(short_line_group[last_idx])
                        merged_lines.append(short_line_group[last_idx + 1])
                else:
                    # If no previous line, handle differently
                    for j in range(min(2, len(short_line_group))):
                        merged_lines.append(short_line_group[j])

                    # Add appropriate removal note
                    if has_mixed_case:
                        merged_lines.append("(Navigation menu removed)")
                    else:
                        merged_lines.append("(Headers removed)")

                    last_idx = len(short_line_group) - 2
                    if last_idx >= 2:
                        merged_lines.append(short_line_group[last_idx])
                        merged_lines.append(short_line_group[last_idx + 1])
            else:
                # For small groups, merge with previous line if possible
                for j, short_line in enumerate(short_line_group):
                    if j == 0 and merged_lines:
                        # First short line gets merged with previous
                        merged_lines[-1] += f". {short_line}"
                    else:
                        # Subsequent lines added separately
                        merged_lines.append(short_line)

        return "\n".join(merged_lines)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on the configured chunk level"""
        chunk_level = self.valves.CHUNK_LEVEL

        # If no chunking requested, return the whole text as a single chunk
        if chunk_level <= 0:
            return [text]

        # Level 1: Phrase-level chunking (split by commas, colons, semicolons)
        if chunk_level == 1:
            # Split by commas, colons, semicolons that are followed by a space
            # First split by newlines to maintain paragraph structure
            paragraphs = text.split("\n")

            # Then split each paragraph by phrases
            chunks = []
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue

                # Split paragraph into phrases
                paragraph_phrases = re.split(r"(?<=[,;:])\s+", paragraph)
                # Only add non-empty phrases
                for phrase in paragraph_phrases:
                    if phrase.strip():
                        chunks.append(phrase.strip())

            return chunks

        # Level 2: Sentence-level chunking (split by periods, exclamation, question marks)
        if chunk_level == 2:
            # Different handling for PDF vs regular content
            if self.is_pdf_content:
                # For PDFs: Don't remove newlines, handle sentences directly
                chunks = []
                # Split by sentences, preserving newlines
                sentences = re.split(r"(?<=[.!?])\s+", text)
                # Only add non-empty sentences
                for sentence in sentences:
                    if sentence.strip():
                        chunks.append(sentence.strip())
            else:
                # For regular content: First split by paragraphs
                paragraphs = text.split("\n")

                chunks = []
                for paragraph in paragraphs:
                    if not paragraph.strip():
                        continue

                    # Split paragraph into sentences
                    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                    # Only add non-empty sentences
                    for sentence in sentences:
                        if sentence.strip():
                            chunks.append(sentence.strip())

            return chunks

        # Level 3: Paragraph-level chunking
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        if chunk_level == 3:
            return paragraphs

        # Level 4-10: Multi-paragraph chunking (4=2 paragraphs, 5=3 paragraphs, etc.)
        chunks = []
        # Calculate how many paragraphs per chunk (chunk_level 4 = 2 paragraphs, 5 = 3 paragraphs, etc.)
        paragraphs_per_chunk = chunk_level - 2

        for i in range(0, len(paragraphs), paragraphs_per_chunk):
            chunk = "\n".join(paragraphs[i : i + paragraphs_per_chunk])
            chunks.append(chunk)

        return chunks
