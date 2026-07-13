The `profiling/` directory hosts utilities for measuring and profiling cdough’s performance.

Contents:

- `stopwatch.h` – Lightweight wall-clock timer.
- `memory.h` – Memory checkpoints for the current process (current/peak resident set size) with stopwatch-like output.
- `thread_profiling.h` – Thread-local CPU usage and timing utilities.
- `output.h` – Storage and formatting of key-value benchmark outputs.
- `utils.h` – Miscellaneous helpers shared across benchmarks.
