"""Claude Haiku calls for Q2: GDELT query generation and zero-shot direction scoring."""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from hms.budget import SpendGuard

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ = 0.10e-6, 0.50e-6, 0.01e-6  # $ per token
CACHE_DIR = Path("data/cache")


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and v.strip():
                os.environ.setdefault(k.strip(), v.strip())


class Query(BaseModel):
    query: str = Field(description="GDELT keyword query, 2-4 terms")


class Direction(BaseModel):
    p_up: float = Field(ge=0.0, le=1.0, description="Probability the YES price rises in 6 h")
    reason: str = Field(description="At most 15 words")


QUERY_SYSTEM = (
    "You write news-search queries for the GDELT DOC API. Given a prediction-market question, "
    "return 2-4 search terms that would find English news headlines relevant to it. Use quoted "
    "phrases for multi-word names (e.g. \"Federal Reserve\"), separate terms with spaces (AND), "
    "no OR, no dates, no site filters, no words shorter than 3 letters. Prefer proper nouns."
)

DIRECTION_SYSTEM = (
    "You are scoring news for a prediction-market research study. You see a Polymarket question "
    "and one news headline that was published just before now. Estimate the probability that the "
    "market's YES price will be HIGHER six hours from now than it is now, considering only the "
    "information in the headline. Markets often have already priced in news by the time a "
    "headline appears, so values near 0.5 are appropriate when the headline is irrelevant, old, "
    "or ambiguous. Return p_up between 0 and 1 and a short reason."
)


class Haiku:
    """Cached, concurrent structured calls. Cache key = hash of (task, system, user text)."""

    def __init__(self, task: str):
        load_key()
        self.client = anthropic.Anthropic(max_retries=6)
        self.path = CACHE_DIR / f"{task}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row
        self.lock = threading.Lock()
        self.budget = SpendGuard()

    def _call(self, system: str, user: str, schema: type[BaseModel]) -> dict:
        key = hashlib.sha256(f"{system}\n{user}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        t = time.perf_counter()
        with self.budget.request() as billed:
            r = self.client.messages.parse(
                model=MODEL,
                max_tokens=2048,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": "low"},
            )
            u = r.usage
            billed[0] = ((u.input_tokens or 0) * PRICE_IN + (u.output_tokens or 0) * PRICE_OUT
                         + (u.cache_read_input_tokens or 0) * PRICE_CACHE_READ
                         + (u.cache_creation_input_tokens or 0) * PRICE_IN * 1.25)
        row = {"key": key, "user": user, "out": r.parsed_output.model_dump(),
               "latency_s": time.perf_counter() - t, "cost_usd": billed[0],
               "in_tok": u.input_tokens, "out_tok": u.output_tokens,
               "cache_read": u.cache_read_input_tokens}
        with self.lock:
            self.cache[key] = row
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    def map(self, system: str, users: list[str], schema: type[BaseModel], workers: int = 8):
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda u: self._call(system, u, schema), users))


def generate_queries(questions: list[str]) -> list[str]:
    rows = Haiku("q2_queries").map(QUERY_SYSTEM, questions, Query)
    return [r["out"]["query"] for r in rows]


def direction_user(question: str, headline: str, with_question: bool = True) -> str:
    if with_question:
        return f"Market question: {question}\nHeadline: {headline}"
    return f"Headline: {headline}"


def score_direction(pairs: list[tuple[str, str]]) -> list[dict]:
    users = [direction_user(q, h) for q, h in pairs]
    return Haiku("q2_direction").map(DIRECTION_SYSTEM, users, Direction)
