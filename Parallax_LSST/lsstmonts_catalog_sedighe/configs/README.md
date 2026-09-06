# Configuration files

Validation configurations are separated into:

- `validation/detectability/`
- `validation/lrt_benchmark/`
- `validation/lrt_history/`

`lrt_benchmark/FAST.json` and `GLOBAL.json` are frozen reference configurations.
Historical development configs under `lrt_history/` document optimizer development and are not the current production setup.

For a new experiment, copy a reference config rather than editing a frozen benchmark config in place.
See the main `README.md` for configuration usage and the current LRT workflow.
