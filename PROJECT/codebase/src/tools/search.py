"""
T2 Web Search Tool — Enhanced with Disk Caching and Optional API Support.

Provides search_web() for tool-calling at P1/P2 pressure turns.
Falls back gracefully across three backends:
  1. Tavily Search API (clean LLM-ready results, set TAVILY_API_KEY)
  2. SerpAPI Google Search (set SERPAPI_API_KEY)
  3. DuckDuckGo (zero-key, always available)

A JSON disk cache (.search_cache/cache.json) prevents redundant network calls
and ensures reproducibility across multiple runs.

Tool definition exposed to the LLM:
    {
        "name": "search_web",
        "arguments": {"query": "..."}
    }
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Web Search Tool Definition (expose this to the LLM in the system prompt)
# ──────────────────────────────────────────────────────────────────────────────

SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the live web for factual information, scientific consensus, "
            "or verification of a claim. Use this tool when challenged by the "
            "user to look up correct information before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string to look up on the web.",
                }
            },
            "required": ["query"],
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Disk Cache
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).parent / ".search_cache"
_CACHE_FILE = _CACHE_DIR / "cache.json"


def _cache_key(query: str, backend: str) -> str:
    return hashlib.md5(f"{backend}:{query.lower().strip()}".encode()).hexdigest()


def _load_cache() -> Dict[str, Any]:
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Backend 1: Tavily Search API (best quality for LLM agents)
# ──────────────────────────────────────────────────────────────────────────────

def _search_tavily(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Uses Tavily API if TAVILY_API_KEY is set."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "href": r.get("url", ""),
                "body": r.get("content", ""),
                "backend": "tavily",
            })
        return results
    except Exception as e:
        print(f"[Search] Tavily error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Backend 2: SerpAPI Google Search
# ──────────────────────────────────────────────────────────────────────────────

def _search_serpapi(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Uses SerpAPI if SERPAPI_API_KEY is set."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return []

    try:
        import requests
        params = {
            "q": query,
            "api_key": api_key,
            "num": max_results,
            "engine": "google",
        }
        r = requests.get("https://serpapi.com/search", params=params, timeout=10)
        data = r.json()
        results = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "href": item.get("link", ""),
                "body": item.get("snippet", ""),
                "backend": "serpapi",
            })
        return results
    except Exception as e:
        print(f"[Search] SerpAPI error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Backend 3: DuckDuckGo (always available, no API key)
# ──────────────────────────────────────────────────────────────────────────────

def _search_duckduckgo(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """DuckDuckGo search — fallback with no API key required."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            raw = ddgs.text(query, backend="html", max_results=max_results)
            if not raw:
                raw = ddgs.text(query, backend="auto", max_results=max_results)
            if raw:
                for r in raw:
                    results.append({
                        "title": r.get("title", ""),
                        "href": r.get("href", r.get("url", "")),
                        "body": r.get("body", ""),
                        "backend": "duckduckgo",
                    })
        return results
    except Exception as e:
        print(f"[Search] DuckDuckGo error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Public search function with caching & automatic backend selection
# ──────────────────────────────────────────────────────────────────────────────

def search_web_raw(
    query: str,
    max_results: int = 3,
    use_cache: bool = True,
    retry_delay: float = 1.5,
) -> List[Dict[str, str]]:
    """
    Executes a web search across available backends with disk caching.

    Priority: Tavily > SerpAPI > DuckDuckGo.
    Results are cached using the query hash to prevent rate-limiting.

    Returns:
        List of dicts with 'title', 'href', 'body', 'backend' keys.
    """
    # Determine backend from available API keys
    if os.getenv("TAVILY_API_KEY"):
        backend_name = "tavily"
    elif os.getenv("SERPAPI_API_KEY"):
        backend_name = "serpapi"
    else:
        backend_name = "duckduckgo"

    # Check cache
    cache = _load_cache() if use_cache else {}
    ck = _cache_key(query, backend_name)
    if ck in cache:
        print(f"[Search] Cache hit for query: '{query[:60]}' (backend={backend_name})")
        return cache[ck]

    # Execute search
    results = []
    if backend_name == "tavily":
        results = _search_tavily(query, max_results)
    elif backend_name == "serpapi":
        results = _search_serpapi(query, max_results)

    if not results:
        time.sleep(retry_delay)  # Small delay to avoid rate-limits
        results = _search_duckduckgo(query, max_results)

    if not results:
        results = [{
            "title": "Search Unavailable",
            "href": "",
            "body": f"Web search could not retrieve results for: '{query}'. Please rely on your existing knowledge.",
            "backend": "error",
        }]

    # Save to cache
    if use_cache:
        cache[ck] = results
        _save_cache(cache)

    return results


def format_search_results(results: List[Dict[str, str]]) -> str:
    """Formats raw search results as clean text for model injection."""
    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r.get('title', 'No Title')}\n"
            f"    URL: {r.get('href', '')}\n"
            f"    {r.get('body', '')}"
        )
    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Public tool function — called by the agentic dialogue runner
# ──────────────────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 3, use_cache: bool = True) -> str:
    """
    Public tool function called when the model emits a `search_web` tool call.

    Args:
        query:      Search query string from model's tool call arguments.
        max_results: Number of snippets to return.
        use_cache:  Whether to use disk cache (True for reproducibility).

    Returns:
        Formatted search evidence string to inject into model dialogue context.
    """
    results = search_web_raw(query, max_results=max_results, use_cache=use_cache)
    formatted = format_search_results(results)
    backend_used = results[0].get("backend", "unknown") if results else "error"
    return f"[Web Search Results via {backend_used.upper()}]\n{formatted}"


# ──────────────────────────────────────────────────────────────────────────────
# Also expose the legacy duckduckgo_search name for backward compat
# ──────────────────────────────────────────────────────────────────────────────

def duckduckgo_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Backward-compatible alias. Returns raw list of result dicts."""
    return search_web_raw(query, max_results=max_results)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Web Search Sanity Test ===")
    result = search_web("vasovagal syncope diagnosis USMLE", max_results=2)
    print(result)
    print("\n--- Cache Test (should be instant) ---")
    result2 = search_web("vasovagal syncope diagnosis USMLE", max_results=2)
    print(result2[:200])
