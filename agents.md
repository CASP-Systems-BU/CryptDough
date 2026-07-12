# Agent Instructions

## Decision Making

- **Always ask for clarification before making important decisions.** If a task is ambiguous, incomplete, or could be interpreted in multiple ways, stop and ask rather than assuming.
- Do not delete, rename, or restructure files without explicit confirmation.
- Do not make architectural or design decisions unilaterally; propose options and let the user decide.
- When unsure about scope (e.g., how much to refactor, which files to touch), ask first.

## Work Pipeline

- When starting a new task, create a Markdown file under `tasks/` with an indexed filename. Example: `0001_adding-agents-file.md`. Check the template in `tasks/0000_task-template.md` to get started.
- Write all relevant information there before coding. Treat it as a small design document.
- Include alternatives with pros and cons, then provide your recommendation.
- Get approval for the task document before proceeding with implementation.
- Update the task document as needed when scope or decisions change.
- Keep `tasks/tasks.md` updated as the short task ledger.
- When a new task document is created, append one short summary line for it at the end of `tasks/tasks.md`.
- When a task status changes, update its line in `tasks/tasks.md` to match the current state.


## Coding Guidelines (Python)

- Follow [PEP 8](https://peps.python.org/pep-0008/) for style and formatting.
- Use type hints on all function signatures.
- Write pure, small functions with a single responsibility.
- Prefer explicit over implicit; avoid clever one-liners that sacrifice readability.
- Use descriptive variable and function names; avoid abbreviations.
- Handle exceptions explicitly; never use bare `except:` clauses.
- Avoid global state; prefer passing values through function arguments.
- Use `pathlib` over `os.path` for file system operations.
- Use `logging` instead of `print` for diagnostic output.
- Add docstrings to public modules, classes, and functions.
- Write tests for non-trivial logic.

## Coding Guidelines (C++)

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) as the primary reference for style and best practices, then defer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) for anything it does not cover.
- Target modern C++ (C++20 or later); prefer standard library facilities over hand-rolled equivalents.
- Use RAII for resource management; avoid manual `new`/`delete`.
- Prefer smart pointers (`std::unique_ptr`, `std::shared_ptr`) over raw owning pointers.
- Use `const` and `constexpr` wherever possible; mark member functions `const` when they do not mutate state.
- Prefer references over pointers when a value must not be null.
- Use `auto` for type deduction only when it improves readability; keep types explicit at interface boundaries.
- Write small, single-responsibility functions and classes.
- Use descriptive names; avoid abbreviations.
- Handle errors explicitly with exceptions or `std::optional`/`std::expected`; never ignore return codes that signal failure.
- Avoid global mutable state; prefer passing values through function arguments.
- Follow the Rule of Zero; fall back to the Rule of Five only when managing resources directly.
- Enable and heed compiler warnings (`-Wall -Wextra -Wpedantic`); treat warnings as errors in CI.
- Use a consistent formatter (e.g., `clang-format`) and a static analyzer (e.g., `clang-tidy`).
- Add documentation comments to public headers, classes, and functions.
- Write tests for non-trivial logic.
