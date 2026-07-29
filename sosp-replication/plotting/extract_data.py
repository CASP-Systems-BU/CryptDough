#!/usr/bin/env python3
"""Extract end-to-end timings from the SOSP artifact-evaluation logs.

The experiment runners under ``sosp-replication/experiments`` append their raw
output to per-configuration log files under ``sosp-replication/data/logs``. This
module parses those logs and regenerates the result CSVs consumed by
``plot_benchmarks.py`` (written to ``data/results``) together with the three
comparison tables (Pigeon, Piranha, MP-SPDZ), which are additionally written to
``data/tables``.

Conventions applied throughout:

* every extracted time is converted to **seconds**;
* when a marker appears multiple times for the same configuration (logs are
  appended with ``>>`` so re-runs accumulate) the runs are **averaged**;
* speed-ups are computed as ``competitor / ours``.
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("extract_data")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
LOGS_DIR = DATA_DIR / "logs"
RESULTS_DIR = DATA_DIR / "results"
TABLES_DIR = DATA_DIR / "tables"

# A signed integer or float, optionally in scientific notation.
_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

THREADS = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------- #
# Generic log-parsing helpers
# --------------------------------------------------------------------------- #
def _read(path: Path) -> str:
    """Return the full text of ``path`` (empty string if it does not exist)."""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _find_all(path: Path, pattern: str, scale: float = 1.0) -> list[float]:
    """Return every ``pattern`` capture-group-1 match in ``path`` as a float.

    ``scale`` multiplies each value (used for millisecond -> second conversion).
    """
    text = _read(path)
    if not text:
        return []
    return [float(m.group(1)) * scale for m in re.finditer(pattern, text)]


def _average(path: Path, pattern: str, scale: float = 1.0) -> Optional[float]:
    """Average every match of ``pattern`` in ``path`` (None if none found)."""
    values = _find_all(path, pattern, scale)
    return mean(values) if values else None


def _slowest(path: Path, pattern: str, scale: float = 1.0) -> Optional[float]:
    """Return the largest (slowest) match of ``pattern`` in ``path``."""
    values = _find_all(path, pattern, scale)
    return max(values) if values else None


def _marker_pattern(label: str) -> str:
    """Pattern for a ``<label> <number> sec`` timing line."""
    return rf"{label}\s+{_NUM}\s+sec"


# --------------------------------------------------------------------------- #
# CSV writing helpers
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float]) -> str:
    """Format a time value with up to 6 significant digits; blank for None."""
    if value is None:
        return ""
    return f"{value:.6g}"


def _speedup(competitor: Optional[float], ours: Optional[float]) -> str:
    """Format ``competitor / ours`` with two decimals; blank if not computable."""
    if competitor is None or not ours:
        return ""
    return f"{competitor / ours:.2f}"


def _write_csv(path: Path, rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    logger.info("  wrote %s", path.relative_to(DATA_DIR.parent))


def _write_block_table(
    filenames: list[Path],
    competitor_label: str,
    blocks: list[tuple[str, str]],
    rows: list[tuple[str, dict[tuple[str, str], tuple[Optional[float], Optional[float]]]]],
    title: Optional[str] = None,
) -> None:
    """Write a comparison table in the multi-header block layout used by the CSVs.

    ``blocks`` is an ordered list of ``(protocol, network)`` pairs; each block
    occupies three columns: ``competitor_label``, ``US`` and ``SpeedUp``. ``rows``
    maps a display label to a per-block ``(competitor, ours)`` pair. The same
    content is written to every path in ``filenames``.
    """
    ncols = 1 + 3 * len(blocks)

    header: list[list[object]] = []
    if title is not None:
        title_row = [""] * ncols
        title_row[0] = title
        header.append(title_row)

    protocol_row = [""] * ncols
    last_protocol: Optional[str] = None
    for index, (protocol, _network) in enumerate(blocks):
        if protocol != last_protocol:
            protocol_row[1 + 3 * index] = protocol
            last_protocol = protocol
    header.append(protocol_row)

    network_row = [""] * ncols
    for index, (_protocol, network) in enumerate(blocks):
        network_row[1 + 3 * index] = network
    header.append(network_row)

    column_row: list[object] = [""] * ncols
    column_row[0] = "Query"
    for index in range(len(blocks)):
        column_row[1 + 3 * index] = competitor_label
        column_row[2 + 3 * index] = "US"
        column_row[3 + 3 * index] = "SpeedUp"
    header.append(column_row)

    data_rows: list[list[object]] = []
    for label, per_block in rows:
        row: list[object] = [""] * ncols
        row[0] = label
        for index, block in enumerate(blocks):
            competitor, ours = per_block.get(block, (None, None))
            row[1 + 3 * index] = _fmt(competitor)
            row[2 + 3 * index] = _fmt(ours)
            row[3 + 3 * index] = _speedup(competitor, ours)
        data_rows.append(row)

    for path in filenames:
        _write_csv(path, header + data_rows)


# --------------------------------------------------------------------------- #
# Figure 5: multi-workload query
# --------------------------------------------------------------------------- #
_FIG5_PROTOCOLS = [("2PC", "2pc"), ("3PC", "3pc"), ("4PC", "4pc"), ("spdz2k", "spdz")]
_FIG5_STAGES = [
    _marker_pattern(r"Stage 1:.*?"),
    _marker_pattern(r"Stage 2:.*?"),
    _marker_pattern(r"Stage 3:.*?"),
    _marker_pattern(r"Stage 4:.*?"),
]


def _multi_workload_total(path: Path) -> Optional[float]:
    """Sum the four (run-averaged) pipeline stages of a multi-workload log."""
    total = 0.0
    for stage_pattern in _FIG5_STAGES:
        stage = _average(path, stage_pattern)
        if stage is None:
            return None
        total += stage
    return total


def extract_fig5() -> None:
    logger.info("Figure 5 (multi-workload)")
    rows: list[list[object]] = [["Protocol", "LAN", "WAN"]]
    for protocol, tag in _FIG5_PROTOCOLS:
        lan = _multi_workload_total(LOGS_DIR / "fig5" / f"lan-{tag}.log")
        wan = _multi_workload_total(LOGS_DIR / "fig5" / f"wan-{tag}.log")
        rows.append([protocol, _fmt(lan), _fmt(wan)])
    _write_csv(RESULTS_DIR / "MULTI_WORKLOAD.csv", rows)


# --------------------------------------------------------------------------- #
# Figure 6: comparison with ORQ
# --------------------------------------------------------------------------- #
_ORQ_QUERIES = [
    ("aspirin", "Aspirin"),
    ("rcdiff", "C. Diff"),
    ("pwd-reuse", "Pwd"),
    ("credit_score", "Credit"),
    ("comorbidity", "Comorbid"),
    ("secrecy_q2", "SecQ2"),
    ("market-share", "Market"),
    ("custom_agg", "SYan"),
    ("distinct_patients", "Patients"),
]
_ORQ_BLOCKS = [("2PC", "LAN"), ("2PC", "WAN"), ("3PC", "LAN"), ("3PC", "WAN")]
_OVERALL = _marker_pattern("Overall")


def _block_tag(protocol: str, network: str) -> tuple[str, str]:
    return network.lower(), protocol.lower()


def extract_fig6() -> None:
    logger.info("Figure 6 (ORQ)")
    cdough_dir = LOGS_DIR / "fig6" / "cdough"
    orq_dir = LOGS_DIR / "fig6" / "orq"

    rows = []
    for stem, label in _ORQ_QUERIES:
        per_block: dict[tuple[str, str], tuple[Optional[float], Optional[float]]] = {}
        for block in _ORQ_BLOCKS:
            net, proto = _block_tag(*block)
            name = f"{net}-{proto}-{stem}.log"
            ours = _average(cdough_dir / name, _OVERALL)
            competitor = _average(orq_dir / name, _OVERALL)
            per_block[block] = (competitor, ours)
        rows.append((label, per_block))

    _write_block_table([RESULTS_DIR / "ORQ.csv"], "ORQ", _ORQ_BLOCKS, rows)


# --------------------------------------------------------------------------- #
# Figure 7: comparison with TVA
# --------------------------------------------------------------------------- #
_TVA_QUERIES = [("energy", "Energy"), ("medical", "mHealth"), ("cloud", "Scheduling")]
_TVA_BLOCKS = [("3PC", "LAN"), ("3PC", "WAN"), ("4PC", "LAN"), ("4PC", "WAN")]
_TVA_ELAPSED = rf"elapsed\s+{_NUM}"


def extract_fig7() -> None:
    logger.info("Figure 7 (TVA)")
    cdough_dir = LOGS_DIR / "fig7" / "cdough"
    tva_dir = LOGS_DIR / "fig7" / "tva"

    rows = []
    for stem, label in _TVA_QUERIES:
        per_block: dict[tuple[str, str], tuple[Optional[float], Optional[float]]] = {}
        for block in _TVA_BLOCKS:
            net, proto = _block_tag(*block)
            name = f"{net}-{proto}-{stem}.log"
            ours = _average(cdough_dir / name, _OVERALL)
            competitor = _average(tva_dir / name, _TVA_ELAPSED)
            per_block[block] = (competitor, ours)
        rows.append((label, per_block))

    _write_block_table(
        [RESULTS_DIR / "TVA.csv"], "TVA", _TVA_BLOCKS, rows, title="Bitonic Sort"
    )


# --------------------------------------------------------------------------- #
# Figure 8: scalability of primitives
# --------------------------------------------------------------------------- #
def _scalability_series(prefix: str, marker: str) -> dict[int, Optional[float]]:
    """Return ``{threads: averaged marker time}`` for a fig8 primitive."""
    series: dict[int, Optional[float]] = {}
    for threads in THREADS:
        path = LOGS_DIR / "fig8" / f"lan-3pc-{prefix}-{threads}t.log"
        series[threads] = _average(path, marker)
    return series


def extract_fig8() -> None:
    logger.info("Figure 8 (scalability)")

    comparison = _scalability_series("micro_sosp_primitives", _marker_pattern(r"\bGR\b"))
    gr_rows: list[list[object]] = [["Network", "Query", "Threads", "Ours"]]
    for threads in THREADS:
        gr_rows.append(["LAN", "GR", threads, _fmt(comparison[threads])])
    _write_csv(RESULTS_DIR / "GR_SCALABILITY.csv", gr_rows)

    rca = _scalability_series("micro_sosp_primitives", _marker_pattern(r"\bRCA\b"))
    ppa = _scalability_series("micro_sosp_primitives", _marker_pattern(r"\bPPA\b"))
    adder_rows: list[list[object]] = [["Network", "Query", "Threads", "Ours"]]
    for threads in THREADS:
        adder_rows.append(["LAN", "RCA", threads, _fmt(rca[threads])])
    for threads in THREADS:
        adder_rows.append(["LAN", "PPA", threads, _fmt(ppa[threads])])
    _write_csv(RESULTS_DIR / "RCA-PPA_SCALABILITY.csv", adder_rows)

    conv2d = _scalability_series("micro_sosp_conv2d", _marker_pattern("Forward"))
    conv_rows: list[list[object]] = [["Threads", "batch -12"]]
    for threads in THREADS:
        conv_rows.append([threads, _fmt(conv2d[threads])])
    _write_csv(RESULTS_DIR / "Conv2D_SCALABILITY.csv", conv_rows)

    bitonic = _scalability_series("micro_sosp_sorting", _marker_pattern("Bitonic"))
    quicksort = _scalability_series("micro_sosp_sorting", _marker_pattern("Quicksort"))
    sort_rows: list[list[object]] = [["Threads", "Network", "Query", "Ours"]]
    for threads in THREADS:
        sort_rows.append([threads, "LAN", "bitonic", _fmt(bitonic[threads])])
    for threads in THREADS:
        sort_rows.append([threads, "LAN", "quicksort", _fmt(quicksort[threads])])
    _write_csv(RESULTS_DIR / "SORTING_SCALABILITY.csv", sort_rows)


# --------------------------------------------------------------------------- #
# Tables 2-4: ML inference comparisons
# --------------------------------------------------------------------------- #
_FORWARD = _marker_pattern("Forward")


def _table_outputs(name: str) -> list[Path]:
    """A table CSV is written to both the results and the tables directories."""
    return [RESULTS_DIR / name, TABLES_DIR / name]


def extract_table2() -> None:
    logger.info("Table 2 (Pigeon)")
    cdough_dir = LOGS_DIR / "table-2" / "cdough"
    pigeon_dir = LOGS_DIR / "table-2" / "pigeon"
    models = [
        ("alexnet", "AlexNet + CIFAR-10"),
        ("vgg16", "VGG16 + CIFAR-10"),
        ("vgg16_imagenet", "VGG16 + ImageNet"),
    ]
    # The paper only reports Pigeon for 3PC; the 4PC block is kept empty so the
    # existing plotting code (which iterates 3PC and 4PC) does not raise.
    blocks = [("3PC", "LAN"), ("3PC", "WAN"), ("4PC", "LAN"), ("4PC", "WAN")]
    pigeon_time = rf"t:\s*{_NUM}\s*s"

    rows = []
    for stem, label in models:
        per_block: dict[tuple[str, str], tuple[Optional[float], Optional[float]]] = {}
        for network in ("LAN", "WAN"):
            name = f"{network.lower()}-3pc-{stem}.log"
            ours = _slowest(cdough_dir / name, _FORWARD)
            competitor = _slowest(pigeon_dir / name, pigeon_time)
            per_block[("3PC", network)] = (competitor, ours)
        rows.append((label, per_block))

    _write_block_table(_table_outputs("Pigeon.csv"), "Pigeon", blocks, rows)


def extract_table3() -> None:
    logger.info("Table 3 (Piranha)")
    cdough_dir = LOGS_DIR / "table-3" / "cdough"
    piranha_dir = LOGS_DIR / "table-3" / "piranha"
    models = [
        ("alexnet", "AlexNet + CIFAR-10"),
        ("vgg16", "VGG16 + CIFAR-10"),
        ("vgg16-imagenet", "VGG16 + ImageNet"),
    ]
    blocks = [("2PC", "LAN"), ("2PC", "WAN")]
    # Piranha reports the inference latency in milliseconds; use party0's log.
    piranha_ms = rf"inference iteration \(ms\),\s*{_NUM}"

    rows = []
    for stem, label in models:
        per_block: dict[tuple[str, str], tuple[Optional[float], Optional[float]]] = {}
        for network in ("LAN", "WAN"):
            name = f"{network.lower()}-2pc-{stem}.log"
            ours = _slowest(cdough_dir / name, _FORWARD)
            competitor = _average(piranha_dir / name, piranha_ms, scale=1e-3)
            per_block[("2PC", network)] = (competitor, ours)
        rows.append((label, per_block))

    _write_block_table(_table_outputs("Piranha.csv"), "Piranha", blocks, rows)


def extract_table4() -> None:
    logger.info("Table 4 (MP-SPDZ)")
    cdough_dir = LOGS_DIR / "table-4" / "cdough"
    mpspdz_dir = LOGS_DIR / "table-4" / "mpspdz"
    blocks = [("spdz2k", "LAN"), ("spdz2k", "WAN")]
    mpspdz_time = rf"Time\s*=\s*{_NUM}\s*seconds"

    per_block: dict[tuple[str, str], tuple[Optional[float], Optional[float]]] = {}
    for network in ("LAN", "WAN"):
        prefix = network.lower()
        ours = _slowest(cdough_dir / f"{prefix}-mpspdz-alexnet.log", _FORWARD)
        competitor = _average(
            mpspdz_dir / f"{prefix}-2pc-alexnet-party0-run1.log", mpspdz_time
        )
        per_block[("spdz2k", network)] = (competitor, ours)

    rows = [("AlexNet", per_block)]
    _write_block_table(_table_outputs("MP-SPDZ.csv"), "mpspdz", blocks, rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    extract_fig5()
    extract_fig6()
    extract_fig7()
    extract_fig8()
    extract_table2()
    extract_table3()
    extract_table4()

    logger.info("Done.")


if __name__ == "__main__":
    main()
