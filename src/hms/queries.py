"""Generate and store one GDELT query per selected market (once; never edited afterwards)."""

from datetime import datetime, timezone

from hms.db import connect
from hms.llm import MODEL, generate_queries


def make_queries() -> int:
    conn = connect()
    todo = conn.execute(
        "SELECT market_id, question FROM markets WHERE selected = 1 AND market_id NOT IN "
        "(SELECT market_id FROM queries) ORDER BY candidate_rank").fetchall()
    if todo:
        now = datetime.now(timezone.utc).isoformat()
        qs = generate_queries([q for _, q in todo])
        conn.executemany("INSERT INTO queries VALUES (?,?,?,?)",
                         [(mid, q, MODEL, now) for (mid, _), q in zip(todo, qs)])
        conn.commit()
    return conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]


if __name__ == "__main__":
    print("queries", make_queries())
