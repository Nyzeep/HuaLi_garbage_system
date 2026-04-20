# huali_garbage_core

Rust core utilities for the HuaLi garbage system.

## Build the HTTP server

```bash
cargo build --release --manifest-path Cargo.toml
```

The HTTP service binary is named `huali_garbage_server`.

## Build the PyO3 extension

```bash
maturin develop --features pyo3
```

## Notes

- `rlib` is used by the HTTP server and Rust unit tests.
- `cdylib` is used by the Python extension module.
- The binary target was renamed to avoid output collisions on Windows.
