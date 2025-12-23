from __future__ import annotations
from src.pipe_shared import *  # noqa: F401,F403

class PipeFetchingMixin:
    async def _rate_limit_domain(self, domain_session_map, domain: str):
        """Apply simple delay to avoid hitting the same domain too fast."""
        if domain not in domain_session_map:
            return
        domain_info = domain_session_map[domain]
        last_access_time = domain_info.get("last_visit", 0)
        current_time = time.time()
        time_since_last_access = current_time - last_access_time

        if time_since_last_access < 3.0:
            base_delay = 2.0
            jitter = random.uniform(0.1, 1.0)
            delay_time = max(0, base_delay - time_since_last_access + jitter)
            if delay_time > 0.1:
                logger.info(
                    f"Rate limiting for domain {domain}: Delaying for {delay_time:.2f} seconds"
                )
                await asyncio.sleep(delay_time)

    async def extract_text_from_html(
        self, html_content: str, prefer_bs4: bool = True
    ) -> str:
        """Extract meaningful text content from HTML."""
        return await self.content_processor.extract_text_from_html(
            html_content, prefer_bs4=prefer_bs4
        )

    async def _fetch_content_impl(self, url: str) -> str:
        """Fetch content from a URL with anti-blocking measures and domain-specific rate limiting"""
        start_time = time.perf_counter() if getattr(self.valves, "DEBUG_TIMING", False) else None
        try:
            state = self.get_state()
            url_considered_count = state.get("url_considered_count", {})
            url_results_cache = state.get("url_results_cache", {})
            master_source_table = state.get("master_source_table", {})
            domain_session_map = state.get("domain_session_map", {})

            # Add to considered URLs counter
            url_considered_count[url] = url_considered_count.get(url, 0) + 1
            self.update_state("url_considered_count", url_considered_count)

            # Check if URL is in cache and use that if available
            if url in url_results_cache:
                logger.info(f"Using cached content for URL: {url}")
                return url_results_cache[url]

            logger.debug(f"Using direct fetch for URL: {url}")

            # Extract domain for session management and tracking
            from urllib.parse import urlparse

            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Domain-specific rate limiting
            await self._rate_limit_domain(domain_session_map, domain)

            # Import fake-useragent for better user agent rotation
            try:
                from fake_useragent import UserAgent

                ua = UserAgent()
                random_user_agent = ua.random
            except ImportError:
                # Fallback if fake-useragent is not installed
                user_agents = [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/123.0.0.0 Safari/537.36",
                ]
                random_user_agent = random.choice(user_agents)

            # Create comprehensive browser fingerprint headers
            headers = self._build_request_headers(domain_session_map, domain)
            chosen_university = headers.get("X-Academic-Institution", "Harvard")
            if domain not in domain_session_map:
                domain_session_map[domain] = {
                    "cookies": {},
                    "last_visit": 0,
                    "visit_count": 0,
                }

            domain_session_map[domain]["cookies"] = {
                "ezproxy_authenticated": "true",
                "institution": chosen_university,
                "access_token": "academic_access_" + str(int(time.time())),
            }

            # Use a mix of academic and standard referrers
            referrers = [
                f"https://library.{chosen_university.lower()}.edu/find/",
                "https://scholar.google.com/scholar?q=",
                "https://www.google.com/search?q=",
                "https://www.bing.com/search?q=",
                "https://search.yahoo.com/search?p=",
                "https://www.scopus.com/record/display.uri",
                "https://www.webofscience.com/wos/woscc/full-record/",
                "https://www.sciencedirect.com/search?",
                "https://www.base-search.net/Search/Results?",
            ]

            # Create rich search terms
            search_terms = [
                parsed_url.path.split("/")[-1].replace(".pdf", "").replace("-", " "),
                (
                    "doi " + parsed_url.path.split("/")[-1]
                    if "/" in parsed_url.path
                    else domain
                ),
                domain + " research",
                domain + " " + query if "query" in locals() else domain,
                query if "query" in locals() else domain + " publication",
            ]

            # Filter out empty or very short ones
            search_terms = [term for term in search_terms if len(term.strip()) > 3]

            # Choose a referrer and term - use hash of domain for consistency while still appearing varied
            domain_hash = hash(domain)
            chosen_referrer = referrers[domain_hash % len(referrers)]
            search_term = search_terms[0] if search_terms else domain
            if len(search_terms) > 1:
                search_term = search_terms[domain_hash % len(search_terms)]

            # Apply the search term
            search_term = search_term.replace(" ", "+")
            headers["Referer"] = chosen_referrer + search_term

            # Update domain tracking info
            if domain not in domain_session_map:
                domain_session_map[domain] = {
                    "cookies": {},
                    "last_visit": 0,
                    "visit_count": 0,
                }

            domain_session = domain_session_map[domain]
            if not isinstance(domain_session, dict):
                domain_session = {"cookies": {}, "last_visit": 0, "visit_count": 0}
                domain_session_map[domain] = domain_session
            domain_session.setdefault("visit_count", 0)
            domain_session["visit_count"] += 1

            domain_session["last_visit"] = time.time()
            self.update_state("domain_session_map", domain_session_map)

            # Create connector with SSL verification disabled and keep session open
            connector = aiohttp.TCPConnector(verify_ssl=False, force_close=True)

            # Check if URL appears to be a PDF
            is_pdf = url.lower().endswith(".pdf")

            # Get existing cookies for this domain if available
            cookie_dict = {}
            if domain in domain_session_map:
                # Convert stored cookies to dictionary format for ClientSession
                stored_cookies = domain_session_map[domain].get("cookies", {})

                # Handle both dictionary and CookieJar formats
                if isinstance(stored_cookies, dict):
                    cookie_dict = stored_cookies
                else:
                    # Try to extract cookies from CookieJar
                    try:
                        for cookie_name, cookie in stored_cookies.items():
                            cookie_dict[cookie_name] = cookie.value
                    except AttributeError:
                        # If that fails, use an empty dict
                        cookie_dict = {}

            async with aiohttp.ClientSession(
                connector=connector, cookies=cookie_dict
            ) as session:
                if is_pdf:
                    # Use binary mode for PDFs
                    async with session.get(
                        url, headers=headers, timeout=20.0
                    ) as response:
                        # Store cookies for future requests
                        if domain in domain_session_map:
                            domain_session_map[domain]["cookies"] = (
                                session.cookie_jar.filter_cookies(url)
                            )
                            self.update_state("domain_session_map", domain_session_map)

                        if response.status == 200:
                            # Get PDF content as bytes
                            pdf_content = await response.read()
                            self.is_pdf_content = True  # Set the PDF flag
                            extracted_content = await self.content_processor.extract_text_from_pdf(
                                pdf_content
                            )

                            # Limit cached content to 3x MAX_RESULT_TOKENS
                            if extracted_content:
                                tokens = await self.count_tokens(extracted_content)
                                token_limit = self.valves.MAX_RESULT_TOKENS * 3
                                if tokens > token_limit:
                                    char_limit = int(
                                        len(extracted_content) * (token_limit / tokens)
                                    )
                                    extracted_content_to_cache = extracted_content[
                                        :char_limit
                                    ]
                                    logger.info(
                                        f"Limiting cached PDF content for URL {url} from {tokens} to {token_limit} tokens"
                                    )
                                else:
                                    extracted_content_to_cache = extracted_content

                                url_results_cache[url] = extracted_content_to_cache
                            else:
                                url_results_cache[url] = extracted_content

                            self.update_state("url_results_cache", url_results_cache)

                            # Add to master source table
                            if url not in master_source_table:
                                title = (
                                    url.split("/")[-1]
                                    .replace(".pdf", "")
                                    .replace("-", " ")
                                    .replace("_", " ")
                                )
                                source_id = f"S{len(master_source_table) + 1}"
                                master_source_table[url] = {
                                    "id": source_id,
                                    "title": title,
                                    "content_preview": extracted_content[:500],
                                    "source_type": "pdf",
                                    "accessed_date": self.research_date,
                                    "cited_in_sections": set(),
                                }
                                self.update_state(
                                    "master_source_table", master_source_table
                                )

                            return extracted_content
                        elif response.status == 403 or response.status == 271:
                            # Try archive.org for 403 errors
                            logger.info(
                                f"Received 403 for PDF {url}, trying archive.org"
                            )
                            archive_content = await self.content_processor.fetch_from_archive(
                                url, session
                            )
                            if archive_content:
                                return archive_content

                            # If archive fallback fails, return original error
                            logger.error(
                                f"Error fetching URL {url}: HTTP {response.status} (archive fallback failed)"
                            )
                            return (
                                f"Error fetching content: HTTP status {response.status}"
                            )
                        else:
                            logger.error(
                                f"Error fetching URL {url}: HTTP {response.status}"
                            )
                            return (
                                f"Error fetching content: HTTP status {response.status}"
                            )
                else:
                    # Normal text/HTML mode
                    async with session.get(
                        url, headers=headers, timeout=20.0
                    ) as response:
                        # Store cookies for future requests
                        if domain in domain_session_map:
                            domain_session_map[domain]["cookies"] = (
                                session.cookie_jar.filter_cookies(url)
                            )
                            self.update_state("domain_session_map", domain_session_map)

                        if response.status == 200:
                            # Check content type in response headers
                            content_type = response.headers.get(
                                "Content-Type", ""
                            ).lower()

                            if "application/pdf" in content_type:
                                # This is a PDF even though the URL didn't end with .pdf
                                pdf_content = await response.read()
                                self.is_pdf_content = True  # Set the PDF flag
                                extracted_content = await self.content_processor.extract_text_from_pdf(
                                    pdf_content
                                )

                                # Limit cached content to 3x MAX_RESULT_TOKENS
                                if extracted_content:
                                    tokens = await self.count_tokens(extracted_content)
                                    token_limit = self.valves.MAX_RESULT_TOKENS * 3
                                    if tokens > token_limit:
                                        char_limit = int(
                                            len(extracted_content)
                                            * (token_limit / tokens)
                                        )
                                        extracted_content_to_cache = extracted_content[
                                            :char_limit
                                        ]
                                        logger.info(
                                            f"Limiting cached PDF content for URL {url} from {tokens} to {token_limit} tokens"
                                        )
                                    else:
                                        extracted_content_to_cache = extracted_content

                                    url_results_cache[url] = extracted_content_to_cache
                                else:
                                    url_results_cache[url] = extracted_content

                                self.update_state(
                                    "url_results_cache", url_results_cache
                                )

                                # Add to master source table
                                if url not in master_source_table:
                                    title = url.split("/")[-1]
                                    if not title or title == "/":
                                        parsed_url = urlparse(url)
                                        title = f"PDF from {parsed_url.netloc}"

                                    source_id = f"S{len(master_source_table) + 1}"
                                    master_source_table[url] = {
                                        "id": source_id,
                                        "title": title,
                                        "content_preview": extracted_content[:500],
                                        "source_type": "pdf",
                                        "accessed_date": self.research_date,
                                        "cited_in_sections": set(),
                                    }
                                    self.update_state(
                                        "master_source_table", master_source_table
                                    )

                                return extracted_content

                            # Handle as normal HTML/text
                            content = await response.text()
                            self.is_pdf_content = False  # Clear the PDF flag
                            if (
                                self.valves.EXTRACT_CONTENT_ONLY
                                and content.strip().startswith("<")
                            ):
                                extracted = await self.content_processor.extract_text_from_html(content)

                                # Limit cached content to 3x MAX_RESULT_TOKENS
                                if extracted:
                                    tokens = await self.count_tokens(extracted)
                                    token_limit = self.valves.MAX_RESULT_TOKENS * 3
                                    if tokens > token_limit:
                                        char_limit = int(
                                            len(extracted) * (token_limit / tokens)
                                        )
                                        extracted_to_cache = extracted[:char_limit]
                                        logger.info(
                                            f"Limiting cached HTML content for URL {url} from {tokens} to {token_limit} tokens"
                                        )
                                    else:
                                        extracted_to_cache = extracted

                                    url_results_cache[url] = extracted_to_cache
                                else:
                                    url_results_cache[url] = extracted

                                self.update_state(
                                    "url_results_cache", url_results_cache
                                )

                                # Add to master source table
                                if url not in master_source_table:
                                    # Try to extract title
                                    title = url
                                    title_match = re.search(
                                        r"<title>(.*?)</title>",
                                        content,
                                        re.IGNORECASE | re.DOTALL,
                                    )
                                    if title_match:
                                        title = title_match.group(1).strip()
                                    else:
                                        # Use domain as title
                                        parsed_url = urlparse(url)
                                        title = parsed_url.netloc

                                    source_id = f"S{len(master_source_table) + 1}"
                                    master_source_table[url] = {
                                        "id": source_id,
                                        "title": title,
                                        "content_preview": extracted[:500],
                                        "source_type": "web",
                                        "accessed_date": self.research_date,
                                        "cited_in_sections": set(),
                                    }
                                    self.update_state(
                                        "master_source_table", master_source_table
                                    )

                                return extracted

                            # Limit cached content to 3x MAX_RESULT_TOKENS
                            if isinstance(content, str):
                                tokens = await self.count_tokens(content)
                                token_limit = self.valves.MAX_RESULT_TOKENS * 3
                                if tokens > token_limit:
                                    char_limit = int(
                                        len(content) * (token_limit / tokens)
                                    )
                                    content_to_cache = content[:char_limit]
                                    logger.info(
                                        f"Limiting cached content for URL {url} from {tokens} to {token_limit} tokens"
                                    )
                                else:
                                    content_to_cache = content

                                url_results_cache[url] = content_to_cache
                            else:
                                url_results_cache[url] = content

                            self.update_state("url_results_cache", url_results_cache)

                            # Add to master source table
                            if url not in master_source_table:
                                # Try to extract title
                                title = url
                                title_match = re.search(
                                    r"<title>(.*?)</title>",
                                    content,
                                    re.IGNORECASE | re.DOTALL,
                                )
                                if title_match:
                                    title = title_match.group(1).strip()
                                else:
                                    # Use domain as title
                                    parsed_url = urlparse(url)
                                    title = parsed_url.netloc

                                source_id = f"S{len(master_source_table) + 1}"
                                master_source_table[url] = {
                                    "id": source_id,
                                    "title": title,
                                    "content_preview": content[:500],
                                    "source_type": "web",
                                    "accessed_date": self.research_date,
                                    "cited_in_sections": set(),
                                }
                                self.update_state(
                                    "master_source_table", master_source_table
                                )

                            return content
                        elif response.status == 403 or response.status == 271:
                            # Try archive.org for 403 errors
                            logger.info(
                                f"Received 403 for URL {url}, trying archive.org"
                            )
                            archive_content = await self.content_processor.fetch_from_archive(
                                url, session
                            )
                            if archive_content:
                                return archive_content

                            # If archive fallback fails, return original error
                            logger.error(
                                f"Error fetching URL {url}: HTTP {response.status} (archive fallback failed)"
                            )
                            return (
                                f"Error fetching content: HTTP status {response.status}"
                            )
                        else:
                            logger.error(
                                f"Error fetching URL {url}: HTTP {response.status}"
                            )
                            return (
                                f"Error fetching content: HTTP status {response.status}"
                            )

        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching content from {url}")
            return f"Timeout while fetching content from {url}"
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error for {url}: {e}")
            return f"Connection error: {str(e)}"
        except aiohttp.ClientOSError as e:
            logger.error(f"OS error for {url}: {e}")
            return f"Connection error: {str(e)}"
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return f"Error fetching content: {str(e)}"
        finally:
            if start_time is not None:
                elapsed = time.perf_counter() - start_time
                logger.info("TIMING fetch:%s %.3fs", url, elapsed)

    async def _fetch_from_archive_impl(self, url: str, session=None) -> str:
        """Fetch content from the Internet Archive (archive.org)"""
        try:
            # Construct Wayback Machine URL
            wayback_api_url = f"https://archive.org/wayback/available?url={url}"

            # Create a new session if not provided
            close_session = False
            if session is None:
                close_session = True
                connector = aiohttp.TCPConnector(verify_ssl=False, force_close=True)
                session = aiohttp.ClientSession(connector=connector)

            try:
                # First check if the URL is archived
                async with session.get(wayback_api_url, timeout=15.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if there are archived snapshots
                        snapshots = data.get("archived_snapshots", {})
                        closest = snapshots.get("closest", {})
                        archived_url = closest.get("url")

                        if archived_url:
                            logger.info(f"Found archive for {url}: {archived_url}")
                            # Fetch the content from the archived URL
                            async with session.get(
                                archived_url, timeout=20.0
                            ) as archive_response:
                                if archive_response.status == 200:
                                    content_type = archive_response.headers.get(
                                        "Content-Type", ""
                                    ).lower()

                                    if "application/pdf" in content_type:
                                        # Handle PDF from archive
                                        pdf_content = await archive_response.read()
                                        self.is_pdf_content = True
                                        extracted_content = (
                                            await self.content_processor.extract_text_from_pdf(
                                                pdf_content
                                            )
                                        )

                                        # Cache the archived content
                                        state = self.get_state()
                                        url_results_cache = state.get(
                                            "url_results_cache", {}
                                        )
                                        url_results_cache[url] = extracted_content
                                        self.update_state(
                                            "url_results_cache", url_results_cache
                                        )

                                        # Update master source table
                                        master_source_table = state.get(
                                            "master_source_table", {}
                                        )
                                        if url not in master_source_table:
                                            title = f"Archived PDF: {url.split('/')[-1].replace('.pdf','').replace('-',' ').replace('_',' ')}"
                                            source_id = (
                                                f"S{len(master_source_table) + 1}"
                                            )
                                            master_source_table[url] = {
                                                "id": source_id,
                                                "title": title,
                                                "content_preview": extracted_content[
                                                    :500
                                                ],
                                                "source_type": "pdf",
                                                "accessed_date": self.research_date,
                                                "cited_in_sections": set(),
                                                "archived": True,
                                            }
                                            self.update_state(
                                                "master_source_table",
                                                master_source_table,
                                            )

                                        return extracted_content
                                    else:
                                        # Handle HTML/text from archive
                                        content = await archive_response.text()
                                        self.is_pdf_content = False

                                        # Extract and clean text if needed
                                        if (
                                            self.valves.EXTRACT_CONTENT_ONLY
                                            and content.strip().startswith("<")
                                        ):
                                            extracted = (
                                                await self.content_processor.extract_text_from_html(
                                                    content
                                                )
                                            )

                                            # Cache the extracted content
                                            state = self.get_state()
                                            url_results_cache = state.get(
                                                "url_results_cache", {}
                                            )
                                            url_results_cache[url] = extracted
                                            self.update_state(
                                                "url_results_cache", url_results_cache
                                            )

                                            # Update master source table
                                            master_source_table = state.get(
                                                "master_source_table", {}
                                            )
                                            if url not in master_source_table:
                                                title = f"Archived: {url}"
                                                title_match = re.search(
                                                    r"<title>(.*?)</title>",
                                                    content,
                                                    re.IGNORECASE | re.DOTALL,
                                                )
                                                if title_match:
                                                    title = f"Archived: {title_match.group(1).strip()}"

                                                source_id = (
                                                    f"S{len(master_source_table) + 1}"
                                                )
                                                master_source_table[url] = {
                                                    "id": source_id,
                                                    "title": title,
                                                    "content_preview": extracted[:500],
                                                    "source_type": "web",
                                                    "accessed_date": self.research_date,
                                                    "cited_in_sections": set(),
                                                    "archived": True,
                                                }
                                                self.update_state(
                                                    "master_source_table",
                                                    master_source_table,
                                                )

                                            return extracted
                                        else:
                                            # Cache the raw content
                                            state = self.get_state()
                                            url_results_cache = state.get(
                                                "url_results_cache", {}
                                            )
                                            url_results_cache[url] = content
                                            self.update_state(
                                                "url_results_cache", url_results_cache
                                            )
                                            return content
                        else:
                            logger.warning(f"No archived version found for {url}")
                            return ""
                    else:
                        logger.warning(
                            f"Error accessing archive.org API: {response.status}"
                        )
                        return ""
            finally:
                # Close the session if we created it
                if close_session and session:
                    await session.close()

        except Exception as e:
            logger.error(f"Error fetching from archive.org: {e}")
            return ""

    async def _extract_text_from_pdf_impl(self, pdf_content) -> str:
        """Extract text from PDF content using PyPDF2 or pdfplumber"""
        if not self.valves.HANDLE_PDFS:
            return "PDF processing is disabled in settings."

        # Ensure we have bytes for the PDF content
        if isinstance(pdf_content, str):
            if pdf_content.startswith("%PDF"):
                pdf_content = pdf_content.encode("utf-8", errors="ignore")
            else:
                return "Error: Invalid PDF content format"

        # Limit extraction to configured max pages to avoid too much processing
        max_pages = self.valves.PDF_MAX_PAGES

        try:
            # Try pypdf / PyPDF2 first
            try:
                import io
                try:
                    from pypdf import PdfReader

                    pdf_backend = "pypdf"
                except ImportError:
                    from PyPDF2 import PdfReader

                    pdf_backend = "PyPDF2"

                # Use ThreadPoolExecutor for CPU-intensive PDF processing
                def extract_with_pypdf():
                    try:
                        # Create a reader object
                        pdf_file = io.BytesIO(pdf_content)
                        pdf_reader = PdfReader(pdf_file)

                        # Get the total number of pages
                        num_pages = len(pdf_reader.pages)
                        logger.info(
                            f"PDF has {num_pages} pages, extracting up to {max_pages}"
                        )

                        # Extract text from each page up to the limit
                        text = []
                        for page_num in range(min(num_pages, max_pages)):
                            try:
                                page = pdf_reader.pages[page_num]
                                page_text = page.extract_text() or ""
                                if page_text.strip():
                                    text.append(f"Page {page_num + 1}:\n{page_text}")
                            except Exception as e:
                                logger.warning(f"Error extracting page {page_num}: {e}")

                        # Join all pages with spacing
                        full_text = "\n\n".join(text)

                        # Add a note if we limited the page count
                        if num_pages > max_pages:
                            full_text += f"\n\n[Note: This PDF has {num_pages} pages, but only the first {max_pages} were processed.]"

                        return full_text if full_text.strip() else None
                    except Exception as e:
                        logger.error(f"Error in PDF extraction with PyPDF2: {e}")
                        return None

                # Execute in thread pool
                loop = asyncio.get_event_loop()
                executor = self._ensure_executor(self._get_current_conversation_id())
                pdf_extract_task = loop.run_in_executor(
                    executor, extract_with_pypdf
                )
                full_text = await pdf_extract_task

                if full_text and full_text.strip():
                    logger.info(
                        f"Successfully extracted text from PDF using {pdf_backend}: {len(full_text)} chars"
                    )
                    return full_text
                else:
                    logger.warning(
                        f"{pdf_backend} extraction returned empty text, trying pdfplumber..."
                    )
            except ImportError as e:
                logger.info(f"Neither pypdf nor PyPDF2 installed ({e}); trying pdfplumber...")
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}, trying pdfplumber...")

            # Try pdfplumber as a fallback
            try:
                import io
                import pdfplumber

                # Use ThreadPoolExecutor for CPU-intensive PDF processing
                def extract_with_pdfplumber():
                    try:
                        pdf_file = io.BytesIO(pdf_content)
                        with pdfplumber.open(pdf_file) as pdf:
                            # Get total pages
                            num_pages = len(pdf.pages)

                            text = []
                            for i, page in enumerate(pdf.pages[:max_pages]):
                                try:
                                    page_text = page.extract_text() or ""
                                    if page_text.strip():
                                        text.append(f"Page {i + 1}:\n{page_text}")
                                except Exception as page_error:
                                    logger.warning(
                                        f"Error extracting page {i} with pdfplumber: {page_error}"
                                    )

                            full_text = "\n\n".join(text)

                            # Add a note if we limited the page count
                            if num_pages > max_pages:
                                full_text += f"\n\n[Note: This PDF has {num_pages} pages, but only the first {max_pages} were processed.]"

                            return full_text
                    except Exception as e:
                        logger.error(f"Error in PDF extraction with pdfplumber: {e}")
                        return None

                # Execute in thread pool
                loop = asyncio.get_event_loop()
                executor = self._ensure_executor(self._get_current_conversation_id())
                pdf_extract_task = loop.run_in_executor(
                    executor, extract_with_pdfplumber
                )
                full_text = await pdf_extract_task

                if full_text and full_text.strip():
                    logger.info(
                        f"Successfully extracted text from PDF using pdfplumber: {len(full_text)} chars"
                    )
                    return full_text
                else:
                    logger.warning("pdfplumber extraction returned empty text")
            except ImportError as e:
                logger.warning(f"pdfplumber not installed: {e}")
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")

            # If both methods failed but we can tell it's a PDF, provide a more useful message
            if pdf_content.startswith(b"%PDF"):
                logger.warning(
                    "PDF detected but text extraction failed. May be scanned or encrypted."
                )
                return "This appears to be a PDF document, but text extraction failed. The PDF may contain scanned images rather than text, or it may be encrypted/protected."

            return "Could not extract text from PDF. The file may not be a valid PDF or may contain security restrictions."

        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return f"Error extracting text from PDF: {str(e)}"

    async def sanitize_query(self, query: str) -> str:
        """Sanitize search query by removing quotes and handling special characters"""
        # Remove quotes that might cause problems with search engines
        sanitized = query.replace('"', " ").replace('"', " ").replace('"', " ")

        # Replace multiple spaces with a single space
        sanitized = " ".join(sanitized.split())

        # Ensure the query isn't too long
        if len(sanitized) > 250:
            sanitized = sanitized[:250]

        logger.info(f"Sanitized query: '{query}' -> '{sanitized}'")
        return sanitized

    def _build_request_headers(self, domain_session_map, domain: str) -> Dict[str, str]:
        """Create realistic headers with rotating user agents."""
        try:
            from fake_useragent import UserAgent

            ua = UserAgent()
            random_user_agent = ua.random
        except ImportError:
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/123.0.0.0 Safari/537.36",
            ]
            random_user_agent = random.choice(user_agents)

        headers = {
            "User-Agent": random_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Chromium";v="116", "Google Chrome";v="116", "Not=A?Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }

        university_ips = {
            "Harvard": "128.103.192." + str(random.randint(1, 254)),
            "Princeton": "128.112.203." + str(random.randint(1, 254)),
            "MIT": "18.7."
            + str(random.randint(1, 254))
            + "."
            + str(random.randint(1, 254)),
            "Stanford": "171.64."
            + str(random.randint(1, 254))
            + "."
            + str(random.randint(1, 254)),
        }

        chosen_university = random.choice(list(university_ips.keys()))
        headers["X-Forwarded-For"] = university_ips[chosen_university]
        headers["X-Requested-With"] = "XMLHttpRequest"
        headers["X-Academic-Institution"] = chosen_university

        if domain not in domain_session_map:
            domain_session_map[domain] = {
                "cookies": {},
                "last_visit": 0,
                "visit_count": 0,
            }
        return headers
