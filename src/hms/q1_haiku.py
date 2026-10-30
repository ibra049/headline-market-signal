"""Q1 zero-shot: Claude Haiku 5.5 with structured JSON output and a cached system prompt."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np

from hms.budget import SpendGuard
from hms.q1_common import (
    LABEL2ID,
    LABELS,
    RESULTS,
    ROOT,
    classification_metrics,
    load_split,
    update_results,
)

MODEL = "claude-haiku-5-5"
# USD per token, Haiku 5.5 (prompts under 100K tokens); cache reads 0.1x, writes 1.25x input.
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6
CACHE_FILE = ROOT / "data" / "cache" / "q1_haiku.jsonl"
CONCURRENCY = 8

SYSTEM = """You are a financial-news sentiment classifier. You read one sentence taken from \
financial news or a company press release and decide how the sentence is likely to affect the \
stock price of the company it is about, from the point of view of an investor.

Labels:
- positive: the sentence reports something that would likely raise the company's share price. \
Examples of such content: revenue, profit, sales or margin growth; a new contract, order or \
customer; an acquisition that strengthens the company; cost savings; raised guidance; a \
dividend increase; market share gains; an upgrade; a successful product launch.
- negative: the sentence reports something that would likely lower the company's share price. \
Examples: falling revenue, profit or sales; operating loss; layoffs presented as a sign of \
weakness; lowered guidance; lost contracts; lawsuits, fines or regulatory trouble; plant \
closures; a downgrade; rising costs that hurt margins; production problems.
- neutral: the sentence is factual or descriptive and gives no clear signal for the share \
price. Examples: a description of what the company does, where it is based, who its managers \
are, the date of an event, the terms of a deal without a judgement, statements that a number \
stayed the same, or background information.

Rules:
1. Judge only the sentence given. Do not use outside knowledge about the company.
2. A sentence comparing two periods is positive if the company's figure improved and negative \
if it worsened, even when the wording is dry ("fell to", "rose to", "decreased", "increased").
3. A smaller loss than before counts as positive; a larger loss counts as negative.
4. If the sentence mixes good and bad news, pick the label of the news that dominates; if \
neither dominates, choose neutral.
5. Many sentences are neutral. Do not force a positive or negative label onto background facts.
6. confidence is your probability, between 0 and 1, that your label is the one a panel of \
financial analysts would choose.

Return only the JSON object required by the schema."""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "confidence": {"type": "number"},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"\'')
                return
    raise SystemExit("ANTHROPIC_API_KEY not found in env or .env")


def key_of(text: str) -> str:
    return hashlib.sha256((MODEL + "\n" + text).encode()).hexdigest()


def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, CACHE_FILE.read_text().splitlines())}


def classify(client: anthropic.Anthropic, text: str, budget: SpendGuard) -> dict:
    t = time.perf_counter()
    with budget.request() as billed:
        row = _classify(client, text, t)
        billed[0] = cost_usd(row)
        return row


def _classify(client, text, t):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "disabled"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"Sentence: {text}"}],
    )
    latency = time.perf_counter() - t
    if resp.stop_reason == "refusal":
        data = {"label": "neutral", "confidence": 1 / 3, "refused": True}
    else:
        data = json.loads(next(b.text for b in resp.content if b.type == "text"))
    u = resp.usage
    return {
        "key": key_of(text),
        "label": data["label"],
        "confidence": float(data["confidence"]),
        "refused": bool(data.get("refused", False)),
        "latency_s": latency,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read": u.cache_read_input_tokens or 0,
        "cache_write": u.cache_creation_input_tokens or 0,
    }


def cost_usd(r: dict) -> float:
    return (r["input_tokens"] * PRICE_IN + r["cache_read"] * PRICE_IN * 0.1
            + r["cache_write"] * PRICE_IN * 1.25 + r["output_tokens"] * PRICE_OUT)


def to_probs(label: str, conf: float) -> np.ndarray:
    """Spread 1 - confidence evenly over the other two labels."""
    conf = min(max(conf, 0.0), 1.0)
    p = np.full(len(LABELS), (1.0 - conf) / (len(LABELS) - 1))
    p[LABEL2ID[label]] = conf
    return p


def main() -> None:
    load_key()
    test = load_split("test")
    texts = test["text"].tolist()
    cache = load_cache()
    todo = [t for t in dict.fromkeys(texts) if key_of(t) not in cache]
    client = anthropic.Anthropic(max_retries=6)
    budget = SpendGuard()
    wall = None
    if todo:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(CONCURRENCY) as pool, CACHE_FILE.open("a") as fh:
            for r in pool.map(lambda x: classify(client, x, budget), todo):
                cache[r["key"]] = r
                fh.write(json.dumps(r) + "\n")
        wall = time.perf_counter() - t0
        wall_meta = {"wall_s": wall, "n": len(todo)}
        (CACHE_FILE.parent / "q1_haiku_wall.json").write_text(json.dumps(wall_meta))
    else:
        meta_path = CACHE_FILE.parent / "q1_haiku_wall.json"
        wall_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

    rows = [cache[key_of(t)] for t in texts]
    probs = np.stack([to_probs(r["label"], r["confidence"]) for r in rows])
    y = test["label"].map(LABEL2ID).to_numpy()
    out = classification_metrics(probs, y)
    n = len(rows)
    out.update({
        "model": MODEL,
        "settings": "thinking disabled, effort low, json_schema output, cached system prompt",
        "n_refused": int(sum(r["refused"] for r in rows)),
        "cost_usd_per_1k": sum(cost_usd(r) for r in rows) / n * 1000,
        "mean_input_tokens": float(np.mean([r["input_tokens"] for r in rows])),
        "mean_cache_read_tokens": float(np.mean([r["cache_read"] for r in rows])),
        "mean_output_tokens": float(np.mean([r["output_tokens"] for r in rows])),
        "median_request_latency_s": float(np.median([r["latency_s"] for r in rows])),
        "latency_s_per_1k_concurrency8": (wall_meta["wall_s"] / wall_meta["n"] * 1000
                                          if wall_meta else None),
        "latency_s_per_1k_sequential_est": float(np.mean([r["latency_s"] for r in rows]) * 1000),
    })
    np.save(RESULTS / "probs_haiku.npy", probs)
    update_results("zeroshot_haiku", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
