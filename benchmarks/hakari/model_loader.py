"""HAKARI custom-loader boundary for the local TypeScript reranker.

No network call is made during import. HAKARI supplies a query and fixed
candidates at invocation time; the endpoint must be loopback-only and explicitly
enabled by the benchmark runner.
"""
from __future__ import annotations
import json
from urllib import request

class LocalTypeScriptReranker:
    def __init__(self, endpoint: str = "http://127.0.0.1:41791/rerank"):
        if not endpoint.startswith("http://127.0.0.1:"):
            raise ValueError("reranker endpoint must be loopback HTTP")
        self.endpoint = endpoint

    def predict(self, pairs, **_kwargs):
        """HAKARI's reranker contract: predict([[query, document], ...])."""
        if not isinstance(pairs, (list, tuple)):
            raise ValueError("pairs must be a sequence")
        if not pairs:
            return []
        query = pairs[0][0]
        if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] != query for pair in pairs):
            raise ValueError("all pairs must contain the same query and one document")
        ranked = self._call_sidecar(query, [{"id": str(index), "text": pair[1]} for index, pair in enumerate(pairs)])
        by_id = {item["id"]: float(item["score"]) for item in ranked}
        return [by_id[str(index)] for index in range(len(pairs))]

    def _call_sidecar(self, query: str, candidates):
        normalized = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("candidate must be an object")
            identifier = candidate.get("id", candidate.get("chunkId"))
            text = candidate.get("text", candidate.get("content"))
            if not isinstance(identifier, str) or not isinstance(text, str):
                raise ValueError("candidate requires id/text")
            normalized.append({"chunkId": identifier, "content": text})
        payload = {"query": query, "candidates": normalized}
        body = json.dumps(payload).encode()
        req = request.Request(self.endpoint, data=body, headers={"content-type": "application/json"})
        with request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict) or not isinstance(result.get("ranked"), list):
            raise ValueError("invalid reranker response")
        return result["ranked"]

    __call__ = predict

def load_model(config=None, **kwargs):
    """HAKARI passes a positional ModelLoadConfig to custom factories."""
    options = dict(getattr(config, 'model_loader_kwargs', None) or {})
    options.update(kwargs)
    return LocalTypeScriptReranker(**options)
