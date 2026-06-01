# codex-usage

An unofficial local token usage reporter for OpenAI Codex CLI/Desktop sessions.

`codex-usage` reads local Codex rollout JSONL files, deduplicates active and
archived copies by session id, and renders a readable terminal table with token
usage and an estimated API-equivalent cost.

It runs entirely locally. It does not call external APIs.

## Features

- Daily, weekly, monthly, and per-session reports.
- Model, reasoning effort, and collaboration mode attribution when present in
  Codex `turn_context` events.
- Rolling windows such as `24h`, `7d`, `30d`, `2w`, and `3m`.
- Explicit date ranges with `--since` and `--until`.
- JSON output for scripting.
- Estimated `Cost (USD)` column using configurable assumptions in the source.
- No runtime dependencies outside Python's standard library.

## Requirements

- Python 3.10 or newer.
- A local Codex install that writes rollout logs under `~/.codex`.

By default the tool reads:

- `~/.codex/sessions`
- `~/.codex/archived_sessions`

Use `--codex-home` if your Codex home is somewhere else.

## Install

Clone the repository:

```sh
git clone https://github.com/<your-org-or-user>/codex-usage.git
cd codex-usage
```

Install it in editable mode:

```sh
python3 -m pip install -e .
```

Run:

```sh
codex-usage
```

You can also run it without installing:

```sh
python3 codex_usage.py
```

## Examples

```sh
# Rolling last 30 days, grouped by day
codex-usage

# Explicit rolling windows
codex-usage --last 7d
codex-usage --last 30d

# Today
codex-usage --today

# Previous calendar week
codex-usage --last-week

# Current calendar week
codex-usage --week

# Current calendar month
codex-usage --month

# Custom inclusive date range
codex-usage --since 2026-05-25 --until 2026-06-01

# Biggest sessions in a rolling window
codex-usage --last 7d --group-by session --limit 10

# Group by week, month, model, reasoning effort, or mode
codex-usage --last 3m --group-by week
codex-usage --last 6m --group-by month
codex-usage --last 7d --group-by model
codex-usage --last 7d --group-by effort
codex-usage --last 7d --group-by mode
codex-usage --last 7d --group-by model-effort

# JSON output
codex-usage --last 7d --json

# Use a custom Codex home
codex-usage --codex-home /path/to/.codex
```

Supported duration units for `--last`: `h`, `d`, `w`, `m`.
Examples: `24h`, `7d`, `30d`, `2w`, `3m`.

## Output Columns

- `Models`: model names from the most recent `turn_context` before each token
  event, such as `gpt-5.5`.
- `Efforts`: reasoning effort from `turn_context`, such as `xhigh`.
- `Modes`: collaboration mode from `turn_context`, such as `default` or `plan`.
- `Input`: total input tokens reported by Codex.
- `Cached Input`: cached input tokens, which are part of `Input`.
- `Uncached`: `Input - Cached Input`.
- `Output`: output tokens reported by Codex.
- `Reasoning`: reasoning output tokens, shown for visibility.
- `Total Tokens`: total tokens reported by Codex.
- `Cost (USD)`: estimated API-equivalent cost.

## Cost Estimate

`Cost (USD)` is an estimate, not a real bill. Local ChatGPT-auth Codex usage is
not API billing.

The current default estimate uses GPT-5.5 API text pricing:

- Uncached input: `$5.00 / 1M tokens`
- Cached input: `$0.50 / 1M tokens`
- Output: `$30.00 / 1M tokens`

Reasoning tokens are shown separately but are treated as part of output tokens,
not billed a second time.

Pricing changes over time. Check the official OpenAI pricing page before using
the estimate for anything important.

## Shell Alias

If you want a shorter command, add your own alias:

```sh
alias cusage="codex-usage"
```

Then reload your shell:

```sh
source ~/.zshrc
```

## Development

Run tests:

```sh
python3 -m unittest discover -s tests
```

Compile check:

```sh
python3 -m py_compile codex_usage.py
```

## Privacy

This tool reads local Codex log files and prints aggregate token usage. It does
not upload logs, send telemetry, or make network requests.

If you share reports publicly, review them first. File paths, session dates, or
session ids may reveal workflow details.

## Limitations

- Codex rollout log formats are not a public stability contract and may change.
- Estimated costs are not billing records.
- Model, effort, and mode are inferred from structured `turn_context` records.
  Older logs without those records will show `unknown`.
- Codex logs do not currently expose a reliable historical service tier per
  token event, so the tool does not claim whether a past event used `fast`.

## License

MIT
