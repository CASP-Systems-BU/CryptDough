The `backend/` directory bridges cdough’s core MPC logic with runtime services such as task scheduling, networking, and memory management.

Sub-directories & files:

- `common/` – General-purpose runtime, task, and networking helpers.
- `null_communicator/` – Null (single-process) service setup.
- `nocopy_communicator/` – TCP-based (no-copy) service setup.
- `service.h` – Umbrella header for all service components.
