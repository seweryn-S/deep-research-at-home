import json
import sys
from datetime import datetime
from typing import Any, Dict, List
import os
import re
import math


def load_chats(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("Expected a JSON array at the top level.")


def extract_messages(chat_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten messages from one chat into a time-ordered list
    with role, content and timestamp.
    """
    history = chat_obj.get("chat", {}).get("history", {})
    messages = history.get("messages", {})
    if isinstance(messages, dict):
        msg_list = list(messages.values())
    elif isinstance(messages, list):
        msg_list = messages
    else:
        return []

    # Sort by timestamp if present to approximate real order
    msg_list = sorted(
        msg_list,
        key=lambda m: m.get("timestamp", 0),
    )

    result: List[Dict[str, Any]] = []
    for m in msg_list:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue

        content = m.get("content") or ""
        # In some exports assistant content is stored in lastSentence
        if not content:
            content = m.get("lastSentence") or ""

        if not content:
            continue

        ts = m.get("timestamp")
        try:
            dt_str = (
                datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                if ts is not None
                else ""
            )
        except (OSError, OverflowError, TypeError):
            dt_str = ""

        result.append(
            {
                "role": role,
                "content": content,
                "timestamp_str": dt_str,
            }
        )

    return result


def build_pdf(chats: List[Dict[str, Any]], output_path: str) -> None:
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Biblioteka 'fpdf2' nie jest zainstalowana.\n"
            "Zainstaluj ją poleceniem:\n"
            "    pip install fpdf2\n"
            "a następnie uruchom skrypt ponownie."
        ) from e

    class NumberedPDF(FPDF):
        def footer(self):
            # Stopka z numerem strony
            self.set_y(-12)
            try:
                self.set_font("DejaVu", "", 9)
            except Exception:
                self.set_font("Helvetica", "", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, f"Strona {self.page_no()}", 0, 0, "C")

    pdf = NumberedPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Konfiguracja czcionek z obsługą polskich znaków
    # W katalogu projektu powinny znajdować się pliki:
    #   DejaVuSans.ttf
    #   DejaVuSans-Bold.ttf  (opcjonalnie, do pogrubienia)
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf")
        try:
            pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf")
        except Exception:
            # Jeśli pogrubiona wersja nie jest dostępna, będziemy używać zwykłej
            pass
    except Exception as e:
        raise SystemExit(
            "Nie udało się wczytać czcionki Unicode.\n"
            "Upewnij się, że w katalogu projektu znajdują się pliki:\n"
            "  DejaVuSans.ttf\n"
            "  (opcjonalnie) DejaVuSans-Bold.ttf\n"
            "Pobierz je np. z oficjalnego pakietu DejaVu Fonts i spróbuj ponownie."
        ) from e

    # Numerowanie stron (tylko bieżący numer)
    # (alias_nb_pages nie jest konieczne, bo nie pokazujemy liczby wszystkich stron)

    for idx, chat_obj in enumerate(chats, start=1):
        pdf.add_page()

        title = chat_obj.get("title") or chat_obj.get("chat", {}).get("title") or ""
        chat_id = chat_obj.get("id", "")

        # Nagłówek czatu
        pdf.set_font("DejaVu", "B", 16)
        header = title or f"Czat #{idx}"
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 10, header)

        if chat_id:
            pdf.set_font("DejaVu", "", 10)
            pdf.ln(2)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 5, f"ID czatu: {chat_id}")

        pdf.ln(5)

        messages = extract_messages(chat_obj)
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            timestamp_str = msg["timestamp_str"]

            # Nagłówek wiadomości
            pdf.set_font("DejaVu", "B", 11)
            header_parts = []
            if timestamp_str:
                header_parts.append(timestamp_str)
            header_parts.append("Użytkownik" if role == "user" else "Asystent")
            header_text = " | ".join(header_parts)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 7, header_text)

            # Treść wiadomości
            pdf.set_font("DejaVu", "", 11)
            # Specjalne traktowanie sekcji pomiędzy
            # "Comprehensive Answer" a "Research conducted on:"
            marker_start = "Comprehensive Answer"
            marker_end = "Research conducted on:"
            if marker_start in content and marker_end in content:
                start_idx = content.index(marker_start)
                end_idx = content.index(marker_end, start_idx)

                before = content[:start_idx]
                middle = content[start_idx:end_idx]
                after = content[end_idx:]

                if before.strip():
                    render_markdown(pdf, before)
                    pdf.ln(3)

                # Wydzielona sekcja na osobnych stronach
                pdf.add_page()
                render_markdown(pdf, middle)
                pdf.ln(3)
                pdf.add_page()

                if after.strip():
                    render_markdown(pdf, after)
            else:
                render_markdown(pdf, content)
            pdf.ln(3)

    # Zapis przez bufor w pamięci – niezależny od implementacji fpdf.output(name)
    pdf_data = pdf.output(dest="S")
    # W zależności od wersji fpdf2 może to być str, bytes lub bytearray
    if isinstance(pdf_data, str):
        pdf_bytes = pdf_data.encode("latin1")
    else:
        pdf_bytes = bytes(pdf_data)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)


def _wrap_long_words(text: str, max_word_length: int = 80) -> str:
    """
    FPDF multi_cell nie potrafi zawijać bardzo długich słów (np. URL-i)
    i rzuca wyjątek „Not enough horizontal space...”.
    Ta funkcja dzieli każde zbyt długie słowo na krótsze fragmenty.
    """

    def wrap_word(word: str) -> str:
        if len(word) <= max_word_length:
            return word
        parts = [
            word[i : i + max_word_length] for i in range(0, len(word), max_word_length)
        ]
        return " ".join(parts)

    wrapped_lines: List[str] = []
    for line in text.splitlines():
        words = line.split(" ")
        wrapped_words = [wrap_word(w) for w in words]
        wrapped_lines.append(" ".join(wrapped_words))

    return "\n".join(wrapped_lines)


def _multi_cell_keep_together(pdf, line_height: float, text: str, indent: float = 0.0):
    """
    Próba utrzymania całego bloku tekstu (akapit/lista/nagłówek) na jednej stronie,
    o ile jego przybliżona wysokość mieści się na stronie.
    """
    if not text:
        return

    text = _wrap_long_words(text)
    lines = text.split("\n")

    effective_width = max(1.0, pdf.epw - indent)

    total_lines = 0
    for line in lines:
        if line == "":
            total_lines += 1
            continue
        line_width = pdf.get_string_width(line)
        n_lines = max(1, math.ceil(line_width / effective_width))
        total_lines += n_lines

    needed_height = total_lines * line_height
    page_height = pdf.h - pdf.t_margin - pdf.b_margin
    remaining = pdf.h - pdf.b_margin - pdf.get_y()

    # Jeśli blok mieści się na pełnej stronie, ale nie w pozostałym miejscu – przenieś na następną
    if needed_height <= page_height and needed_height > remaining:
        pdf.add_page()

    pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(effective_width, line_height, text)


def _inline_md_to_plain(text: str) -> str:
    """
    Prosta obsługa formatowania inline Markdown:
    Zamiast kasować znaczniki, zamieniamy je na prostą, widoczną formę:
    - **pogrubienie**, __pogrubienie__ -> TEKST WIELKIMI LITERAMI
    - *kursywa*, _kursywa_              -> _tekst_ (otoczony podkreślnikami)
    - `kod`                             -> «kod»
    - [tekst](url)                      -> tekst [url]
    """
    # Linki [tekst](url) -> 'tekst [url]'
    def repl_link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        return f"{label} [{url}]"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl_link, text)

    # Pogrubienie **...** albo __...__ -> WIELKIE LITERY
    def repl_bold(m: re.Match) -> str:
        inner = m.group(2)
        return inner.upper()

    text = re.sub(r"(\*\*|__)(.+?)\1", repl_bold, text)

    # Kursywa *...* albo _..._ -> otoczona podkreślnikami
    def repl_italic(m: re.Match) -> str:
        inner = m.group(1)
        return f"_{inner}_"

    # Najpierw *...* niebędące częścią **...**
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", repl_italic, text)
    # Potem _..._
    text = re.sub(r"_(.+?)_", repl_italic, text)

    # Kod inline `...` -> «...»
    def repl_code(m: re.Match) -> str:
        inner = m.group(1)
        return f"«{inner}»"

    text = re.sub(r"`([^`]+)`", repl_code, text)

    return text


def _parse_inline_md_segments(text: str):
    """
    Zamienia tekst z prostym Markdownem inline na listę segmentów:
    [(tekst, styl), ...] gdzie styl w {"plain","bold","italic","code"}.
    """
    # Najpierw uprość linki: [tekst](url) -> 'tekst [url]'
    def repl_link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        return f"{label} [{url}]"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl_link, text)

    segments: List[tuple[str, str]] = []
    buf: List[str] = []
    bold = False
    italic = False
    code = False
    i = 0
    n = len(text)

    def flush():
        if not buf:
            return
        seg_text = "".join(buf)
        buf.clear()
        if code:
            style = "code"
        elif bold and italic:
            style = "bold"  # brak osobnego stylu B+I, traktujemy jako pogrubione
        elif bold:
            style = "bold"
        elif italic:
            style = "italic"
        else:
            style = "plain"
        segments.append((seg_text, style))

    while i < n:
        ch = text[i]

        # Kod inline `
        if ch == "`":
            flush()
            code = not code
            i += 1
            continue

        if code:
            buf.append(ch)
            i += 1
            continue

        # Pogrubienie ** albo __
        if text.startswith("**", i) or text.startswith("__", i):
            flush()
            bold = not bold
            i += 2
            continue

        # Kursywa * lub _
        if ch in ("*", "_"):
            # Gwiazdki podwójne są obsługiwane wyżej jako **,
            # tutaj traktujemy pojedyncze * oraz _ jako przełącznik kursywy
            flush()
            italic = not italic
            i += 1
            continue

        buf.append(ch)
        i += 1

    flush()
    return segments


def _write_inline_paragraph(
    pdf, text: str, line_height: float = 6.0, prefix: str = ""
):
    """
    Renderuje pojedynczy wiersz z prostym Markdownem inline,
    zachowując pogrubienie/kursywę/kod w ramach linii.
    """
    text = _wrap_long_words(text)

    # Generujemy segmenty
    segments = _parse_inline_md_segments(text)
    if prefix:
        segments.insert(0, (prefix, "plain"))

    first_segment = True
    for seg_text, style in segments:
        if not seg_text:
            continue

        font_style = ""
        if style == "bold":
            font_style = "B"

        # Kod – wyróżniamy cudzysłowami kątowymi
        if style == "code":
            seg_text = f"«{seg_text}»"

        # Nie zmieniamy aktualnego rozmiaru czcionki – korzystamy z tego,
        # który został ustawiony przed wywołaniem tej funkcji.
        current_size = getattr(pdf, "font_size_pt", 11)
        try:
            pdf.set_font("DejaVu", font_style, current_size)
        except Exception:
            pdf.set_font("DejaVu", "", current_size)

        if first_segment:
            pdf.set_x(pdf.l_margin)
            first_segment = False

        pdf.write(line_height, seg_text)

    pdf.ln(line_height)


def render_markdown(pdf, text: str) -> None:
    """
    Bardzo prosty renderer Markdown:
    - #, ##, ###  -> nagłówki o różnej wielkości
    - -, *, +     -> listy wypunktowane (•)
    - 1. 2. ...   -> listy numerowane
    - ```         -> bloki kodu (monospacowane / wcięte)
    Pozostałe linie traktowane jako zwykłe akapity.
    """

    lines = text.splitlines()
    in_code_block = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # Pusty wiersz -> odstęp
        if not line.strip():
            pdf.ln(3)
            continue

        # Przełączanie bloków kodu
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            pdf.ln(1)
            continue

        if in_code_block:
            # Blok kodu – użyj tej samej czcionki, ale z wcięciem
            _multi_cell_keep_together(pdf, 5, line, indent=4)
            continue

        # Cała linia pogrubiona w stylu Markdown: **Tekst**
        bold_line = re.match(r"^\s*\*\*(.+?)\*\*\s*$", line)
        if bold_line:
            content = bold_line.group(1).strip()
            pdf.set_font("DejaVu", "B", 11)
            _multi_cell_keep_together(pdf, 6, content)
            pdf.set_font("DejaVu", "", 11)
            pdf.ln(1)
            continue

        # Cała linia kursywą w stylu Markdown: *Tekst*
        italic_line = re.match(r"^\s*\*(.+?)\*\s*$", line)
        if italic_line:
            content = italic_line.group(1).strip()
            # Nie mamy zarejestrowanej wersji italic, więc traktujemy to jak zwykły akapit.
            _multi_cell_keep_together(pdf, 6, content)
            pdf.ln(1)
            continue

        # Nagłówki Markdown
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            level = len(heading_match.group(1))
            content = heading_match.group(2).strip()
            # Zmniejszaj czcionkę wraz z poziomem nagłówka
            base_size = 16
            size = max(10, base_size - (level - 1) * 2)
            pdf.set_font("DejaVu", "B", size)
            _multi_cell_keep_together(pdf, size / 2 + 3, content)
            pdf.set_font("DejaVu", "", 11)
            pdf.ln(1)
            continue

        # Listy wypunktowane
        bullet_match = re.match(r"^[-*+]\s+(.*)", line)
        if bullet_match:
            content = bullet_match.group(1).strip()
            _write_inline_paragraph(pdf, content, line_height=6, prefix="• ")
            continue

        # Listy numerowane
        ordered_match = re.match(r"^\d+\.\s+(.*)", line)
        if ordered_match:
            content = ordered_match.group(0).strip()
            _write_inline_paragraph(pdf, content, line_height=6)
            continue

        # Zwykły tekst / akapit
        _write_inline_paragraph(pdf, line, line_height=6)


def main(argv: List[str]) -> None:
    if len(argv) < 3:
        print(
            "Użycie:\n"
            "  python export_chat_pdf.py chat-export-...json output.pdf\n"
        )
        raise SystemExit(1)

    input_path = argv[1]
    output_path = argv[2]

    chats = load_chats(input_path)
    build_pdf(chats, output_path)
    abs_out = os.path.abspath(output_path)
    print(f"Zapisano PDF: {abs_out}")
    if not os.path.exists(abs_out):
        print(
            "UWAGA: Plik nie został znaleziony pod powyższą ścieżką.\n"
            "Sprawdź proszę, z jakiego katalogu uruchamiasz skrypt (polecenie `pwd`)."
        )


if __name__ == "__main__":
    main(sys.argv)
