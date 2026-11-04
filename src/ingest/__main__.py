"""Ingestion CLI.

Phases 1-5 support load, normalize, chunk, embed, and index.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import load_config, validate_runtime
from src.ingest.chunk import run_chunk
from src.ingest.embed import run_embed
from src.ingest.loader import run_load
from src.ingest.normalize import run_normalize
from src.store.chroma import run_index

IMPLEMENTED_STAGES = {"load", "normalize", "chunk", "embed", "index"}


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HDFC MF Facts ingestion pipeline")
    parser.add_argument(
        "--stage",
        default="load",
        choices=["load", "normalize", "chunk", "embed", "index", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--inventory", default="sources.csv", help="Path to sources.csv")
    parser.add_argument("--only", default="", help="Comma-separated source_id filter")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list sources only")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first load failure")
    parser.add_argument("--rebuild", action="store_true", help="Force embed/index rebuild")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        warnings = validate_runtime(config)
    except Exception as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    only = [item for item in args.only.split(",") if item.strip()] or None

    if args.stage in {"load", "all"}:
        try:
            result = run_load(
                config,
                inventory_path=args.inventory,
                only=only,
                dry_run=args.dry_run,
                fail_fast=args.fail_fast,
            )
        except Exception as exc:
            print(f"INGEST ERROR: {exc}", file=sys.stderr)
            return 1

        if result.get("dry_run"):
            _print(result)
        else:
            _print(
                {
                    "manifest_path": result.get("manifest_path"),
                    "inventory": result.get("inventory"),
                    "summary": result.get("summary"),
                    "sources": [
                        {
                            "source_id": item["source_id"],
                            "status": item["status"],
                            "http_status": item["http_status"],
                            "bytes": item["bytes"],
                            "local_path": item["local_path"],
                        }
                        for item in result.get("sources", [])
                    ],
                }
            )

    if args.stage in {"normalize", "all"}:
        try:
            result = run_normalize(
                config,
                inventory_path=args.inventory,
                only=only,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"NORMALIZE ERROR: {exc}", file=sys.stderr)
            return 1

        if result.get("dry_run"):
            _print(result)
        else:
            _print(
                {
                    "coverage_path": result.get("coverage_path"),
                    "summary": result.get("summary"),
                    "sources": result.get("sources"),
                }
            )

    if args.stage in {"chunk", "all"}:
        try:
            result = run_chunk(
                config,
                only=only,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"CHUNK ERROR: {exc}", file=sys.stderr)
            return 1

        if result.get("dry_run"):
            _print(result)
        else:
            _print(result)

    if args.stage in {"embed", "all"}:
        if args.dry_run:
            _print({"dry_run": True, "note": "Embedding is skipped in dry-run mode."})
        else:
            try:
                result = run_embed(config, force=args.rebuild)
            except Exception as exc:
                print(f"EMBED ERROR: {exc}", file=sys.stderr)
                return 1
            _print(result)

    if args.stage in {"index", "all"}:
        if args.dry_run:
            _print({"dry_run": True, "note": "Indexing is skipped in dry-run mode."})
        else:
            try:
                result = run_index(config, force=args.rebuild)
            except Exception as exc:
                print(f"INDEX ERROR: {exc}", file=sys.stderr)
                return 1
            _print(result)

    if args.stage not in IMPLEMENTED_STAGES:
        print(
            f"STAGE NOT IMPLEMENTED: {args.stage} arrives in a later phase.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
