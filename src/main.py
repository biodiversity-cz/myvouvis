"""Entry: python src/main.py [herbarium-dwc] — batch bot or scheduler service."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bots.implementations.herbarium_dwc_databot import HerbariumDwcDatabot  # noqa: E402
from config import config  # noqa: E402
from core.application.BotScheduler import BotScheduler  # noqa: E402
from core.application.JobStore import JobStore  # noqa: E402
from core.application.WorkerPool import WorkerPool  # noqa: E402
from web.app import BotUI  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bot", nargs="?", help="Run only this bot (e.g. herbarium-dwc)")
    args = parser.parse_args()

    available = {HerbariumDwcDatabot.NAME: HerbariumDwcDatabot}

    if args.bot:
        if args.bot not in available:
            print(f"Unknown bot: {args.bot}. Available: {', '.join(available)}", file=sys.stderr)
            sys.exit(1)
        available[args.bot]().run()
        return

    job_queue: queue.Queue = queue.Queue()
    job_store = JobStore()
    WorkerPool(job_queue, job_store).start()
    scheduler = BotScheduler(job_queue, available)
    scheduler.start()

    ui = BotUI(job_store, scheduler)
    threading.Thread(
        target=lambda: ui.run(
            host="0.0.0.0",
            port=int(config.get_application_config("port", 5000)),
        ),
        daemon=True,
    ).start()

    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        scheduler.stop()


if __name__ == "__main__":
    main()
