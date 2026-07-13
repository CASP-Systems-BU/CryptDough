The `matrix/` directory provides matrix containers built on top of cdough's vector types.

Contents:

- `matrix.h` – Umbrella header exposing the `PlainMatrix`, `SecureMatrix`, and `HeightWidth` aliases.
- `hybrid/` – Concrete matrix implementations:
    - `matrix.h` – Base `Matrix` class and shared element-/matrix-wise operator definitions.
    - `plain_matrix.h` – Plaintext matrix implementation.
    - `secure_matrix.h` – Secret-shared (secure) matrix implementation.
