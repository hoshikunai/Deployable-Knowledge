from __future__ import annotations

import json
from urllib.parse import urlparse
from typing import Any
from urllib import error, parse, request


class DeployableKnowledgeClient:
    def __init__(
        self,
        base_url: str,
        search_timeout: float = 180,
        ingestion_timeout: float = 1800,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base URL must be an absolute HTTP(S) URL")
        if "[" in base_url or "](" in base_url:
            raise ValueError("base URL must not be Markdown-formatted")
        self.base_url = base_url.rstrip("/")
        self.search_timeout = search_timeout
        self.ingestion_timeout = ingestion_timeout

    def ingest_document(self, title: str, text: str) -> dict[str, Any]:
        payload = json.dumps(
            {
                "title": title,
                "text": text,
            }
        ).encode("utf-8")

        http_request = request.Request(
            f"{self.base_url}/documents",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        completed_result: dict[str, Any] | None = None

        try:
            with request.urlopen(http_request, timeout=self.ingestion_timeout) as response:
                # The endpoint emits one JSON object per line.
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    event = json.loads(line)
                    status = event.get("status")

                    if status == "error":
                        raise RuntimeError(
                            event.get("message", "Document ingestion failed")
                        )

                    if status == "complete":
                        completed_result = event["result"]
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Document ingestion returned HTTP {exc.code}: {body}"
            ) from exc

        if completed_result is None:
            raise RuntimeError("Ingestion ended without a completion event")

        return completed_result

    def search(self, query: str, top_k: int) -> dict[str, Any]:
        parameters = parse.urlencode(
            {
                "query": query,
                "topK": top_k,
            }
        )

        http_request = request.Request(
            f"{self.base_url}/search?{parameters}",
            method="GET",
        )

        try:
            with request.urlopen(http_request, timeout=self.search_timeout) as response:
                return json.load(response)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Search returned HTTP {exc.code}: {body}"
            ) from exc

    def heartbeat(self) -> None:
        try:
            with request.urlopen(f"{self.base_url}/heartbeat", timeout=30):
                return
        except error.HTTPError as exc:
            raise RuntimeError(
                f"Application heartbeat returned HTTP {exc.code}"
            ) from exc
