#!/usr/bin/env python3
"""Run salon runtime service without GUI (for diagnostics)."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.salon_ai_system import SalonAISystem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run salon runtime pipeline")
    p.add_argument("--config", default="data/config/salon_ai.runtime.yaml")
    p.add_argument("--print-every-sec", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    system = SalonAISystem(config_path=args.config)
    stop = {"flag": False}

    def _shutdown(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    system.start()
    try:
        while not stop["flag"]:
            time.sleep(max(0.2, float(args.print_every_sec)))
            status = system.status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
    finally:
        system.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
