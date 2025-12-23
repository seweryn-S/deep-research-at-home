import asyncio
import logging
import re
import aiohttp
import random
import time
from urllib.parse import urlparse

logger = logging.getLogger("Deep Research at Home")

class ContentProcessor:
    """Handles content fetching and extraction for the Pipe."""
    def __init__(self, pipe):
        self._pipe = pipe
    def __getattr__(self, name):
        return getattr(self._pipe, name)
    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL with anti-blocking measures and caching."""
        return await self._pipe._fetch_content_impl(url)
    async def extract_text_from_html(self, html_content: str, prefer_bs4: bool = True) -> str:
        """Extract meaningful text content from HTML."""
        from src.constants import NAVIGATION_CLASS_PATTERNS
        try:
            def _regex_extract(raw_html: str) -> str:
                import html
                content = html.unescape(raw_html)
                content = re.sub(r"<(script|style|head|nav|header|footer)[^>]*>.*?</\1>", " ", content, flags=re.DOTALL)
                content = re.sub(r"<[^>]*>", " ", content)
                content = re.sub(r"\.([A-Z])", ". \\1", content)
                return re.sub(r"\s+", " ", content).strip()
            try:
                from bs4 import BeautifulSoup
                import html
                unescaped = html.unescape(html_content)
                soup = BeautifulSoup(unescaped, "html.parser")
                for tag in ["script", "style", "head", "iframe", "noscript", "nav", "header", "footer", "aside", "form"]:
                    for element in soup(tag): element.decompose()
                for element in soup.find_all(class_=lambda c: c and any(x.lower() in c.lower() for x in NAVIGATION_CLASS_PATTERNS)):
                    element.decompose()
                text = soup.get_text(" ", strip=True)
                text = re.sub(r" {2,}", " ", text)
                text = re.sub(r"\.([A-Z])", ". \\1", text)
                return "\n\n".join([line.strip() for line in text.split("\n") if line.strip()])
            except Exception:
                return _regex_extract(html_content)
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return html_content
    async def extract_text_from_pdf(self, pdf_content) -> str:
        """Extract text from PDF content."""
        try:
            # Keep a single implementation of PDF extraction logic in Pipe.
            return await self._pipe._extract_text_from_pdf_impl(pdf_content)
        except AttributeError:
            if not self.valves.HANDLE_PDFS:
                return "PDF processing is disabled."
            return "PDF processing is unavailable in this build."
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")
            return "Could not extract text from PDF."
    async def fetch_from_archive(self, url: str, session=None) -> str:
        """Fetch content from the Internet Archive."""
        try:
            wayback_url = f"https://archive.org/wayback/available?url={url}"
            close_session = False
            if session is None:
                close_session = True
                session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
            try:
                async with session.get(wayback_url, timeout=15.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        archived_url = data.get("archived_snapshots", {}).get("closest", {}).get("url")
                        if archived_url:
                            async with session.get(archived_url, timeout=20.0) as aresp:
                                if aresp.status == 200: return await aresp.text()
                return ""
            finally:
                if close_session: await session.close()
        except Exception: return ""
