"""Run a command and emit datasource-sync notification if datasource surface changed."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from animal_breed.sync.datasource_sync import DatasourceSyncTransaction


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for datasource surface scanning.",
    )
    parser.add_argument(
        "--origin-session-id",
        default=os.getenv("CODEX_THREAD_ID", "unknown_session"),
        help="Origin session id included in the payload.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute (prefix with --).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("No command provided. Use: animal-datasource-sync-notify -- <command...>", file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root).resolve()
    with DatasourceSyncTransaction(repo_root=repo_root, origin_session_id=args.origin_session_id):
        completed = subprocess.run(command, cwd=str(repo_root))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

