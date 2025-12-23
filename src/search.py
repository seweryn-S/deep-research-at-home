import asyncio
import json
import logging
import aiohttp
from typing import List, Dict, Optional
from src.utils.logger import setup_logger

logger = logging.getLogger("Deep Research at Home")

class SearchClient:
    """Handles web search responsibilities for the Pipe."""

    def __init__(self, pipe):
        self._pipe = pipe

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    async def search_web(self, query: str) -> List[Dict]:
        """Perform web search using OpenWebUI first, then fallback."""
        logger.debug(f"Starting web search for query: {query}")
        
        results = await self._try_openwebui_search(query)
        if not results:
            logger.debug(f"OpenWebUI search returned no results, trying fallback search for: {query}")
            results = await self._fallback_search(query)

        if results:
            return results

        logger.warning(f"No search results found for query: {query}")
        return [{
            "title": f"No results for '{query}'",
            "url": "",
            "snippet": f"No search results were found for the query: {query}",
        }]

    async def _try_openwebui_search(self, query: str) -> List[Dict]:
        """Try to use Open WebUI's built-in search functionality."""
        try:
            from open_webui.routers.retrieval import process_web_search, SearchForm
            search_form = SearchForm(queries=[query])
            search_task = asyncio.create_task(
                process_web_search(self.__request__, search_form, user=self.__user__)
            )
            search_results = await asyncio.wait_for(search_task, timeout=15.0)
            
            state = self.get_state()
            url_selected_count = state.get("url_selected_count", {})
            repeat_count = sum(1 for url, count in url_selected_count.items() if count >= self.valves.REPEATS_BEFORE_EXPANSION)
            total_results = self.valves.SEARCH_RESULTS_PER_QUERY + self.valves.EXTRA_RESULTS_PER_QUERY + min(repeat_count, self.valves.EXTRA_RESULTS_PER_QUERY)

            results: List[Dict] = []
            if search_results:
                if "docs" in search_results:
                    docs = search_results.get("docs", [])
                    urls = search_results.get("filenames", [])
                    for i, doc in enumerate(docs[:total_results]):
                        results.append({"title": f"'{query}'", "url": urls[i] if i < len(urls) else "", "snippet": doc})
                elif "collection_name" in search_results:
                    collection_name = search_results.get("collection_name")
                    urls = search_results.get("filenames", [])
                    for i, url in enumerate(urls[:total_results]):
                        results.append({"title": f"Search Result {i+1} from {collection_name}", "url": url, "snippet": f"Result from collection: {collection_name}"})
            return results
        except Exception as e:
            logger.error(f"Error in _try_openwebui_search: {str(e)}")
            return []

    async def _fallback_search(self, query: str) -> List[Dict]:
        """Fallback search using direct HTTP request."""
        try:
            from urllib.parse import quote
            encoded_query = quote(query)
            search_url = f"{self.valves.SEARCH_URL}{encoded_query}"
            
            state = self.get_state()
            url_selected_count = state.get("url_selected_count", {})
            repeat_count = sum(1 for url, count in url_selected_count.items() if count >= self.valves.REPEATS_BEFORE_EXPANSION)
            total_results = self.valves.SEARCH_RESULTS_PER_QUERY + self.valves.EXTRA_RESULTS_PER_QUERY + min(repeat_count, self.valves.EXTRA_RESULTS_PER_QUERY)

            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(search_url, timeout=15.0) as response:
                    if response.status != 200: return []
                    content_type = response.headers.get("Content-Type", "").lower()
                    text = await response.text()
                    
                    if "application/json" in content_type:
                        data = json.loads(text)
                        items = data.get("results") or data.get("items") or data.get("data") or []
                        return [{"title": item.get("title") or item.get("name") or f"Result {i+1}", "url": item.get("url") or item.get("link") or "", "snippet": item.get("snippet") or item.get("content") or item.get("description") or ""} for i, item in enumerate(items[:total_results])]

                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(text, "html.parser")
                        result_elements = soup.select("article.result")
                        results = []
                        for i, element in enumerate(result_elements[:total_results]):
                            title_element = element.select_one("h3 a")
                            url_element = element.select_one("h3 a")
                            snippet_element = element.select_one("p.content")
                            results.append({
                                "title": title_element.get_text() if title_element else f"Result {i+1}",
                                "url": url_element.get("href") if url_element else "",
                                "snippet": snippet_element.get_text() if snippet_element else ""
                            })
                        return results
                    except Exception: return []
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []
