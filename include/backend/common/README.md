The `common/` directory provides general runtime facilities shared by all cdough services.

Contents:

- `setup.h` – Parses command-line flags and sets up the runtime environment.
- `rand_setup.h` – Randomness setup helpers.
- `setting.h` – Network setting (`SAME`/`LAN`/`WAN`) enum and parser.
- `libote_io.h` – Shared Boost.Asio I/O context used by libOTe-backed correlations.
- `task.h` – Work unit abstraction.
- `worker.h` – Worker thread implementation.
- `runtime.h` – Central runtime object orchestrating tasks, memory, and communication.
- `util.h` – General runtime helpers (e.g. unsigned-type selection).
