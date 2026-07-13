The `operators/` directory implements MPC-enabled data-parallel operators used by CryptDough queries.

Contents:

- `operators.h` – Umbrella header aggregating all operator headers.
- `aggregation_selector.h` - Internal helper class to select between aggregation types at runtime
- `aggregation.h` – Secure aggregation functions (SUM, COUNT, etc.) and oblivious aggregation network
- `circuits.h` - Implementation of various boolean circuits.
- `common.h` – Common utilities shared by operators.
- `distinct.h` – Distinct operator.
- `join.h` - Join operator.
- `machine_learning.h` – Secure machine-learning operators (e.g. weight loading, matrix-based inference).
- `merge.h` – Oblivious Merge.
- `prefix_network.h` – Base class for prefix (scan) networks over shared vectors.
- `quicksort.h` – Quicksort implementation.
- `radixsort.h` – Radixsort implementation.
- `shuffle.h` – Oblivious shuffle operators.
- `sorting.h` – Sorting helper functions.
- `sorting_network.h` – Pairwise sorting-network building blocks.
- `streaming.h` – Streaming operators.
