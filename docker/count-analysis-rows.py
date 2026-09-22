#!/usr/bin/env python3
"""Count the rows one owner contributes to an `mpc-analysis` run.

Each organization holds its own half of `cdrcatsse_match_pcc`. The run manifest
has to declare how many rows that half contributes, because both halves are
padded to one common length and a party cannot read the length of a file it is
not allowed to see.

The declared count is NOT the file's line count. `flagged_dx` pass 1 keeps only
Emergency and Acute Care encounters, so what the owner contributes is the number
that survives that filter. This script applies it and prints the number.

It is deliberately standalone: `mpc-analysis` connects to its peers during
startup, and the counts are needed *before* the manifest -- and therefore the
ports -- are agreed. Nothing here touches the network, and nothing leaves the
machine except the single integer the operator puts in the manifest.

Usage:
    ./docker/count-analysis-rows.py /srv/org-a/private/base_owner_a.csv
    ./docker/count-analysis-rows.py --verbose base_owner_a.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

# flagged_dx pass 1: WHERE visit_type_pcc IN ('Emergency', 'Acute Care').
# Must match kKeptVisitTypes in playground/etl.h exactly.
KEPT_VISIT_TYPES: frozenset[str] = frozenset({"Emergency", "Acute Care"})

VISIT_TYPE_COLUMN = "visit_type_pcc"

logger = logging.getLogger(__name__)


def count_kept_rows(path: Path) -> tuple[int, Counter[str]]:
    """Return the post-filter row count and a tally of every visit type seen.

    Args:
        path: A CSV holding one owner's half of `cdrcatsse_match_pcc`.

    Returns:
        The number of rows surviving the pass-1 visit-type filter, and a
        Counter of the raw `visit_type_pcc` values, for spotting a value that
        should have been kept but was spelled differently.

    Raises:
        KeyError: If the file has no `visit_type_pcc` column.
    """
    kept = 0
    seen: Counter[str] = Counter()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or VISIT_TYPE_COLUMN not in reader.fieldnames:
            raise KeyError(
                f"{path} has no {VISIT_TYPE_COLUMN} column; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            visit_type = (row.get(VISIT_TYPE_COLUMN) or "").strip()
            seen[visit_type] += 1
            if visit_type in KEPT_VISIT_TYPES:
                kept += 1

    return kept, seen


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Count the rows one owner contributes to an mpc-analysis run, "
            "for the --rows-a / --rows-b manifest entries."
        )
    )
    parser.add_argument("csv_path", type=Path, help="this owner's half of cdrcatsse_match_pcc")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also report the visit types that were dropped",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stderr)

    if not args.csv_path.is_file():
        logger.error("no such file: %s", args.csv_path)
        return 1

    try:
        kept, seen = count_kept_rows(args.csv_path)
    except KeyError as exc:
        logger.error("%s", exc)
        return 1
    except UnicodeDecodeError as exc:
        logger.error("%s is not UTF-8: %s", args.csv_path, exc)
        return 1

    if args.verbose:
        total = sum(seen.values())
        logger.info("%s: %d rows, %d kept", args.csv_path, total, kept)
        for visit_type, count in seen.most_common():
            mark = "keep" if visit_type in KEPT_VISIT_TYPES else "drop"
            logger.info("  %-4s %-30s %d", mark, visit_type or "(empty)", count)
        # A visit type that differs only by case or spacing is the likeliest
        # cause of a count the other parties do not expect.
        near = [
            v
            for v in seen
            if v not in KEPT_VISIT_TYPES
            and v.strip().casefold() in {k.casefold() for k in KEPT_VISIT_TYPES}
        ]
        if near:
            logger.warning("visit types differing only by case or spacing: %s", near)

    # Only the number goes to stdout, so this composes in a shell pipeline.
    print(kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
