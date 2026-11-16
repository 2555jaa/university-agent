"""Browser and academic web inspection tools."""

import re
import logging
from typing import Any, Dict, Optional
import httpx

from app.security.audit import audit_logger

logger = logging.getLogger("university_agent.tools.browser")


class BrowserTools:
    """Tools for fetching academic papers, documentation, and web references."""

    @staticmethod
    def _strip_html(html_text: str) -> str:
        """Strip HTML tags and scripts to produce readable text."""
        # Remove scripts and style blocks
        clean = re.sub(r'<script.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<style.*?</style>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        # Convert paragraphs / breaks to newlines
        clean = re.sub(r'<(p|br|div|h[1-6]|li)\b[^>]*>', '\n', clean, flags=re.IGNORECASE)
        # Strip all remaining tags
        clean = re.sub(r'<[^>]+>', '', clean)
        # Clean up whitespace
        clean = re.sub(r'[ \t]+', ' ', clean)
        clean = re.sub(r'\n\s*\n', '\n\n', clean)
        return clean.strip()

    @classmethod
    async def fetch_academic_url(cls, url: str) -> Dict[str, Any]:
        """Fetch article or documentation from approved HTTP/HTTPS URL."""
        if not (url.startswith("http://") or url.startswith("https://")):
            return {"status": "error", "error": "URL must start with http:// or https://"}

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                headers = {"User-Agent": "UniversityAIAgent/1.0 (Academic Research Assistant)"}
                res = await client.get(url, headers=headers)
                
                if res.status_code != 200:
                    return {"status": "error", "status_code": res.status_code, "error": f"Failed with HTTP {res.status_code}"}

                content_type = res.headers.get("content-type", "").lower()
                title_match = re.search(r'<title>(.*?)</title>', res.text, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else url

                text_content = cls._strip_html(res.text)

                audit_logger.log_event(
                    event_type="browser",
                    action="fetch_url",
                    details={"url": url, "title": title, "length": len(text_content)},
                    risk_level="low",
                    status="success"
                )

                return {
                    "status": "success",
                    "url": url,
                    "title": title,
                    "content": text_content[:5000],  # Scoped preview
                    "total_length": len(text_content)
                }
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return {"status": "error", "error": str(e)}


browser_tools = BrowserTools()
