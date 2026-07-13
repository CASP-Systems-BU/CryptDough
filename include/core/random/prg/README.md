The `prg/` directory provides local and shared pseudorandom generator (PRG) interfaces and
implementations.

Contents:

- `random_generator.h` – Abstract base class for random number generators.
- `prg_algorithm.h` – PRG algorithm interfaces and implementations (e.g. AES, dev/urandom, XChaCha20).
- `common_prg.h` – Common (shared-seed) PRG shared across a group of parties, plus the `CommonPRGManager`.
- `seeded_prg.h` – PRG implementation built on `CommonPRG` with a random seed.
- `committed_seeds_queue.h` – Commitment protocol for fresh, batched seed generation.
- `zero_rg.h` – Random generator that outputs all zeros (for testing and benchmarking).
