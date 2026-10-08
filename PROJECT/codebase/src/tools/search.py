"""
DuckDuckGo Web Search Tool Module.
Provides standalone search capability and standard function definition for LLM tool-calling.
"""

from typing import List, Dict, Any

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "duckduckgo_search",
        "description": "Search the live web for factual information, current events, verification, or answers using DuckDuckGo.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string to search on the web.",
                }
            },
            "required": ["query"],
        },
    },
}


def duckduckgo_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Executes a web search via DuckDuckGo without requiring an API key.

    Args:
        query: Search query string.
        max_results: Maximum number of search snippets to return.

    Returns:
        List of dictionaries with 'title', 'href', and 'body'.
    """
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            # First try html backend which reliably works across network configurations
            raw_results = ddgs.text(query, backend="html", max_results=max_results)
            if not raw_results:
                raw_results = ddgs.text(query, backend="auto", max_results=max_results)
            if not raw_results:
                raw_news = ddgs.news(query, max_results=max_results)
                if raw_news:
                    raw_results = [
                        {"title": n.get("title", ""), "href": n.get("url", ""), "body": n.get("body", "")}
                        for n in raw_news
                    ]

            if raw_results:
                for r in raw_results:
                    results.append({
                        "title": r.get("title", ""),
                        "href": r.get("href", r.get("url", "")),
                        "body": r.get("body", ""),
                    })
    except Exception as e:
        # Fallback to news if text throws an error
        try:
            with DDGS() as ddgs:
                raw_news = ddgs.news(query, max_results=max_results)
                if raw_news:
                    for n in raw_news:
                        results.append({
                            "title": n.get("title", ""),
                            "href": n.get("url", ""),
                            "body": n.get("body", ""),
                        })
        except Exception as e2:
            results.append({
                "title": "Search Error",
                "href": "",
                "body": f"Failed to retrieve web search results: {str(e2)}",
            })
    return results


def format_search_results(results: List[Dict[str, str]]) -> str:
    """Formats raw search results into a clean string for LLM injection."""
    if not results:
        return "No search results found."
    
    formatted_chunks = []
    for i, r in enumerate(results, 1):
        formatted_chunks.append(
            f"[{i}] Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}"
        )
    return "\n\n".join(formatted_chunks)


if __name__ == "__main__":
    print("Testing DuckDuckGo Search...")
    test_query = "latest space exploration news"
    res = duckduckgo_search(test_query, max_results=3)
    print(f"Results for query: '{test_query}':\n")
    print(format_search_results(res))
