"""CLI: python -m bebop.batch {plan,submit,status,collect} ...
Run from the bebop tree with $PREFECT_API_URL reachable. JSON on stdout."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import driver
from .crops import find_pdfs
from .state import default_state_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bebop.batch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, default=default_state_dir() / "default",
                    help="state dir (state.json + chunk files); default cache/batch/default")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _pdf_args(p):
        p.add_argument("--pdfs", nargs="*", default=[], help="PDF files and/or directories (recursive)")
        p.add_argument("--pdf-list", type=Path, help="text file, one PDF path per line")
        p.add_argument("--chunk-mb", type=int, default=driver.DEFAULT_CHUNK_BYTES >> 20,
                       help="roll to a new request file past this many MiB (Files API cap: 2 GB)")

    _pdf_args(sub.add_parser("plan", help="enumerate crops, write request files, estimate cost"))
    s = sub.add_parser("submit", help="upload + create batch jobs for unsubmitted chunks")
    _pdf_args(s)
    s.add_argument("--dry-run", action="store_true", help="everything but the API calls")
    sub.add_parser("status", help="refresh and report job states")
    c = sub.add_parser("collect", help="write succeeded jobs' responses into the read cache")
    c.add_argument("--wait", type=float, default=0.0, help="hours to keep polling for completion")
    c.add_argument("--poll", type=float, default=60.0, help="seconds between polls")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    state_dir: Path = args.state
    state = driver.load_or_fresh(state_dir)

    report: dict = {}
    if args.cmd in ("plan", "submit"):
        roots = list(args.pdfs)
        if args.pdf_list:
            roots += [ln.strip() for ln in args.pdf_list.read_text().splitlines() if ln.strip()]
        if roots:
            report["plan"] = driver.plan(find_pdfs(roots), state, state_dir,
                                         chunk_bytes=args.chunk_mb << 20)
        elif args.cmd == "plan":
            ap.error("plan needs --pdfs and/or --pdf-list")
    if args.cmd == "submit":
        report["submit"] = driver.submit(state, state_dir, _api(), dry_run=args.dry_run)
    elif args.cmd == "status":
        report = driver.status(state, state_dir, _api())
    elif args.cmd == "collect":
        report = driver.collect(state, state_dir, _api(), wait_s=args.wait * 3600, poll_s=args.poll)
    print(json.dumps(report, indent=1))
    return 0


def _api():
    from .gemini import GeminiBatchApi

    return GeminiBatchApi()


if __name__ == "__main__":
    sys.exit(main())
