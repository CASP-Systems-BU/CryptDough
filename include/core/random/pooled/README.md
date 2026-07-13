The `pooled/` directory provides a pooled randomness wrapper that separates correlation
generation from retrieval.

Contents:

- `pooled_generator.h` – Wraps another correlation generator and pools pre-generated correlations in a queue for efficient batch retrieval.
