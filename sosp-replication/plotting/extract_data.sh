#!/bin/bash
#
# Extract end-to-end timings from the artifact-evaluation logs
# (sosp-replication/data/logs) into the result CSVs consumed by
# plot_benchmarks.py (sosp-replication/data/results) and the comparison-table
# CSVs (sosp-replication/data/tables).
#
# This is a thin wrapper around extract_data.py; see that file for the parsing
# rules (units are normalised to seconds, repeated runs are averaged, and
# speed-ups are computed as competitor / ours).

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

# Prefer the project virtual environment (.env, created from requirements.txt)
# when it exists so the run is reproducible; otherwise fall back to system python3.
if [[ -x "$repo_root/.env/bin/python3" ]]; then
    python_bin="$repo_root/.env/bin/python3"
else
    python_bin="python3"
fi

"$python_bin" "$script_dir/extract_data.py" "$@"
