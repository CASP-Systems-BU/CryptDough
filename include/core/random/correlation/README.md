The `correlation/` directory provides the correlation generators (OLE, OT, triples, OPRF, and
related primitives) consumed by cdough's protocols.

Contents:

- `correlation_generator.h` – Abstract base class for all correlation generators.
- `ole_generator.h` – Abstract base class for Oblivious Linear Evaluation (OLE) generators.
- `ot_generator.h` – Oblivious Transfer (OT) generator built on the OLE interface.
- `gilboa_ole.h` – Gilboa Oblivious Linear Evaluation generator.
- `gilboa_mod_p.h` – Gilboa OLE modulo a small prime.
- `gilboa_crt.h` – Gilboa OLE composed over multiple primes via CRT.
- `silent_ot.h` – Silent OT generator (libOTe-backed).
- `dpf.h` – Distributed Point Function (DPF) generator.
- `oprf.h` – Oblivious Pseudorandom Function (OPRF) implementation.
- `beaver_triple_generator.h` – Generates Beaver multiplication triples.
- `dummy_auth_triple_generator.h` – Base classes for authenticated triple generators.
- `dummy_auth_random_generator.h` – Insecure authenticated random generator for testing.
- `dummy_ole.h` – Insecure dummy OLE generator for testing.
- `zero_ole.h` – OLE/OT generator that outputs all zeros (for testing and benchmarking).
- `zero_sharing_generator.h` – Generates sharings of zero for arithmetic and binary operations.
- `shprg.h` – Seed-Homomorphic Pseudorandom Generator (SHPRG).
- `libsecjoin.h` – Correlations used with libSecureJoin (DPF/OPRF), with a mock fallback.
- `mock.h` – Empty stand-ins for DPF/OPRF when libSecureJoin is unavailable.
- `registry.h` – Type-indexed registry of correlation generators.
