# Contributing

Thanks for considering a contribution.

## Local Setup

```sh
python3 -m pip install -e .
python3 -m unittest discover -s tests
```

## Development Notes

- Keep runtime dependencies at zero unless a dependency clearly improves the
  user experience enough to justify installation friction.
- Do not commit real Codex rollout logs.
- Do not commit screenshots, local paths, usernames, API keys, or private
  account details.
- Add tests for parser, aggregation, and output changes.

## Verification

Before opening a pull request, run:

```sh
python3 -m unittest discover -s tests
python3 -m py_compile codex_usage.py
```
