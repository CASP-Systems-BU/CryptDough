The `protocols/` directory contains the MPC protocol implementations supported by CryptDough.

Contents:

- `interface/` - has abstract classes that define the protocol API.
- `eda_bits.h` – Extended doubly-authenticated bits (edaBits) protocol wrapper.
- `op_structs.h` – Operator functors (arithmetic/boolean) shared across protocol implementations.
- `dummy_0pc.h` – Dummy 0-party protocol (mock only, not functional, useful for testing).
- `plaintext_1pc.h` – Plaintext 1-party protocol (no privacy, useful for testing and debugging, but could be used inside of a TEE).
- `beaver_2pc.h` – Dishonest-majority 2-party protocol using Beaver triples.
- `replicated_3pc.h` – Replicated secret sharing 3-party protocol.
- `dalskov_4pc.h` – Dalskov et al. Fantastic-Four 4-party protocol.
- `custom_4pc.h` – Rewrite of Fantastic 4PC to be round efficient.
- `spdz2k_npc.h` - Dishonest-majority n-party malicious protocol.