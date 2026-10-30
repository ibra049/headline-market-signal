"""Shared laptop resource guard for sequential training runs."""

import fcntl
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def training_guard():
    path = Path("artifacts/training.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training process holds artifacts/training.lock") from exc
        if sys.platform == "darwin":
            result = subprocess.run(["memory_pressure"], check=True, capture_output=True, text=True)
            match = re.search(r"System-wide memory free percentage:\s*(\d+)%", result.stdout)
            if match is None or int(match[1]) < 20:
                raise RuntimeError("Training requires memory_pressure free >= 20%")
            print(f"memory_pressure free: {match[1]}%", flush=True)
        yield
