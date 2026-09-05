"""LLM judge via OpenRouter (OpenAI-compatible chat API, stdlib only).

Key lives in .env (OPENROUTER_API_KEY), never committed. Verdicts are strict
JSON; cost/accounting comes from the API's own usage fields.
"""

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# $/token, from https://openrouter.ai/api/v1/models (checked 2026-09): update if stale
PRICES_PER_1M = {
    "z-ai/glm-5.3-flash": (0.075, 0.25),
    "openai/gpt-4o-mini": (0.15, 0.60),
}
GLM_FLASH = "z-ai/glm-5.3-flash"
GPT4O_MINI = "openai/gpt-4o-mini"

VERDICTS = ("faithful", "unfaithful", "unverifiable")

SYSTEM_PROMPT = (
    "You are a RAG faithfulness judge. / Du bist ein RAG-Treuerichter.\n"
    "Decide if the CLAIM is fully supported by the CONTEXT alone (never use "
    "outside knowledge). Answer with ONLY one JSON object, no other text:\n"
    '{"verdict": "faithful"} - every fact in the claim is supported by the context\n'
    '{"verdict": "unfaithful"} - at least one fact contradicts or is unsupported '
    "by the context (wrong entity, number, date, or fabricated detail)\n"
    '{"verdict": "unverifiable"} - the context does not address the claim at all'
)


def load_api_key(env_path=".env") -> str:
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key
    try:
        for line in open(env_path):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    raise RuntimeError("OPENROUTER_API_KEY not in env or .env")


def _verdict_key(model: str, context: str, claim: str) -> str:
    return hashlib.sha256(f"{model}\x00{context}\x00{claim}".encode()).hexdigest()


def _parse_verdict(text: str) -> str:
    m = re.search(r'"verdict"\s*:\s*"(\w+)"', text)
    verdict = m.group(1) if m else ""
    if verdict not in VERDICTS:
        raise ValueError(f"unparseable verdict: {text[:200]!r}")
    return verdict


@dataclass
class CostLog:
    calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    usd: float = 0.0
    by_model: dict = field(default_factory=dict)
    parse_errors: int = 0


class OpenRouterJudge:
    """Chat-model judge. checkpoint = OpenRouter model id."""

    def __init__(
        self,
        model: str = GLM_FLASH,
        api_key: str | None = None,
        cost_log: CostLog | None = None,
        max_tokens: int = 256,
        cache_path: Path | None = None,
    ):
        self.checkpoint = model
        self.api_key = api_key or load_api_key()
        self.max_tokens = max_tokens
        self.log = cost_log if cost_log is not None else CostLog()
        self.price_in, self.price_out = PRICES_PER_1M.get(model, (0.0, 0.0))
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._vcache: dict[str, str] = {}
        if cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._vcache[row["key"]] = row["verdict"]

    def _call(self, user_msg: str, retries: int = 3, max_tokens: int | None = None) -> dict:
        body = json.dumps(
            {
                "model": self.checkpoint,
                "temperature": 0,
                "max_tokens": max_tokens or self.max_tokens,
                "reasoning": {"exclude": True},  # hide reasoning; still billed ~100-250 tok
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            }
        ).encode()
        for attempt in range(retries):
            req = urllib.request.Request(
                API_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError("unreachable: retry loop exhausted")

    def _account(self, resp: dict) -> None:
        with self._lock:  # concurrent arm-B threads
            self._account_locked(resp)

    def _account_locked(self, resp: dict) -> None:
        usage = resp.get("usage") or {}
        tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        # prefer the API's own billing figure; fall back to our price table
        usd = (
            float(usage["cost"])
            if usage.get("cost") is not None
            else (tin * self.price_in + tout * self.price_out) / 1_000_000
        )
        self.log.calls += 1
        self.log.in_tokens += tin
        self.log.out_tokens += tout
        self.log.usd += usd
        m = self.log.by_model.setdefault(
            self.checkpoint, {"calls": 0, "in": 0, "out": 0, "usd": 0.0}
        )
        m["calls"] += 1
        m["in"] += tin
        m["out"] += tout
        m["usd"] += usd

    def verdict(self, context: str, claim: str) -> str:
        key = _verdict_key(self.checkpoint, context, claim)
        with self._lock:
            if key in self._vcache:
                return self._vcache[key]
        msg = f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}"
        for max_tokens in (self.max_tokens, self.max_tokens * 2):
            resp = self._call(msg, max_tokens=max_tokens)
            self._account(resp)
            content = resp["choices"][0]["message"].get("content")
            try:
                verdict = _parse_verdict(content or "")
                self._store(key, verdict)
                return verdict
            except ValueError:
                continue
        self.log.parse_errors += 1
        verdict = "unverifiable"  # parse failure: recorded in cost log, conservative
        self._store(key, verdict)
        return verdict

    def _store(self, key: str, verdict: str) -> None:
        with self._lock:
            self._vcache[key] = verdict
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")
