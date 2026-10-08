# codex-usage

An unofficial local token usage reporter for OpenAI Codex CLI/Desktop sessions.

This project is a fork of [hashmil/codex-usage](https://github.com/hashmil/codex-usage).

`codex-usage` reads local Codex rollout JSONL files, deduplicates active and
archived copies by session id, and renders a readable terminal table with token
usage and an estimated API-equivalent cost.

Logs are processed locally. Published prices are fetched from the official OpenAI
documentation at most once per day; no logs are sent. Use `--offline` to disable
network access. No API key or pricing server is needed.

## Features

- Daily, weekly, monthly, and per-session reports.
- Model, reasoning effort, and collaboration mode attribution when present in
  Codex `turn_context` events.
- Terminal-width-aware tables with multiline model, effort, and mode cells.
- Rolling windows such as `24h`, `7d`, `30d`, `2w`, and `3m`.
- Explicit date ranges with `--since` and `--until`.
- JSON output for scripting.
- Model-specific published API text prices, long-context rates, and recorded price history.
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
git clone https://github.com/volodymyrkorobkin/codex-usage.git
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

# Force every token column, even on narrower terminals
codex-usage --table full

# Force the narrower summary table
codex-usage --table compact

# Test or screenshot a specific width
codex-usage --width 100

# Use a custom Codex home
codex-usage --codex-home /path/to/.codex
```

Supported duration units for `--last`: `h`, `d`, `w`, `m`.
Examples: `24h`, `7d`, `30d`, `2w`, `3m`.

## Output Columns

The default `--table auto` mode detects the terminal width. It uses the full
table when it fits and falls back to a compact table when needed. `Models`,
`Efforts`, and `Modes` can wrap across multiple lines inside a row.

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

## API-equivalent Cost

`Cost (USD)` is an estimate, not a real bill. Local ChatGPT-auth Codex usage is
not API billing.

The tool reads the labelled text pricing tables from the official
[OpenAI pricing page](https://developers.openai.com/api/docs/pricing.md). Each
usage event is priced using its model before aggregation, including model
switches within a session. Reasoning is part of output and is never charged a
second time. The default processing tier is **Standard**; Codex collaboration
mode and reasoning effort are not processing tiers.

For example, the published short-context Standard rates checked on October 8,
2026 are (USD per million tokens):

| Model | Uncached input | Cached input | Output |
| --- | ---: | ---: | ---: |
| gpt-6-luna | 0.10 | 0.01 | 0.50 |
| gpt-6-sol | 2.00 | 0.20 | 10.00 |
| gpt-6.1-sol | 2.00 | 0.10 | 10.00 |

Prompts above 272,000 input tokens use the published long-context rates for the
whole request. Request size comes from `last_token_usage`, when it matches the
cumulative usage delta. If that information is absent or the delta covers
multiple requests, short-context rates are assumed and the report says so.
Cache writes, tool fees, regional premiums, and other account-specific charges
are not included because rollout token totals do not identify them reliably.

```sh
# Same command as before; fetches/caches official prices automatically
codex-usage --last-week --group-by model --table full

# Refresh now, or run entirely offline
codex-usage --refresh-prices
codex-usage --offline

# Explicit processing-tier assumption for the entire report
codex-usage --pricing-tier fast

# Require a recorded price period covering each event
codex-usage --strict-pricing-history

# Use your own price history without fetching anything
codex-usage --pricing-file prices.json
```

### Price history and unavailable costs

Prices are cached in `~/.cache/codex-usage/pricing.json` (or under
`XDG_CACHE_HOME`; Windows uses `LOCALAPPDATA`). Refreshes preserve old price
records and append changed prices with `start` inclusive and `end` exclusive,
in UTC. An unchanged price keeps its original start date. The bundled snapshot
provides prices when the cache or network is unavailable. Refresh failures warn
on stderr and the report includes the last successful check date.

The official page supplies **current prices, not historical effective dates**.
Automatically collected periods have `date_basis: "observed"`: their start/end
represent when this tool detected a change, which can be later than the actual
change. For events before available history, the default uses the latest
published model price and explicitly labels that fallback. `--strict-pricing-history`
disables that fallback. It does not turn observation dates into verified dates.

Models without an exact published price, unsupported cache/context rates, or
uncovered dates in strict mode have no estimated price. Mixed rows and the total
still show the cost of priced events with an asterisk, such as `$12.34*`. The
asterisk means **partial cost: unpriced events are excluded**. A row with no priced
events shows `N/A`. All events remain included in token counts.

`codex-auto-review` is a reviewer label found in Codex logs. Codex Auto-review
checks eligible tool approval requests (see the
[official documentation](https://learn.chatgpt.com/docs/cyber-safety/recommended-configuration)).
The pricing source does not give this label an API rate, so the tool keeps its
usage visible and unpriced rather than assigning another model's rate.

JSON output includes `pricing.rates` with the model, tier, source, recorded
start/end, date basis, and rates actually selected, plus pricing notes and
`unpriced_events` in rows and totals. `known_cost_usd` contains the priced
subtotal and `priced_events` counts the events contributing to it.
`estimated_cost_usd` remains `null` when any events are unpriced, so scripts can
distinguish a complete estimate from a partial subtotal.

A supplied `--pricing-file` uses the same schema. For independently verified
historical periods, use `date_basis: "effective"` and cite the evidence in
`source`. Dates must include a timezone; periods for a model/tier cannot overlap.
A minimal catalog looks like this (the dates below illustrate the schema):

```json
{
  "schema_version": 1,
  "checked_at": "2026-10-08T00:00:00+00:00",
  "prices": [
    {
      "model": "gpt-6-sol",
      "tier": "standard",
      "start": "2026-10-08T00:00:00+00:00",
      "end": null,
      "date_basis": "observed",
      "source": "https://developers.openai.com/api/docs/pricing",
      "long_context_threshold": 272000,
      "rates": {
        "input": 2.0, "cached_input": 0.2, "cache_write": 2.5, "output": 10.0,
        "long_input": 4.0, "long_cached_input": 0.4,
        "long_cache_write": 5.0, "long_output": 15.0
      }
    }
  ]
}
```


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
not upload logs or send telemetry. Price refreshes make an HTTPS GET to the
official public documentation; `--offline` and `--pricing-file` disable that
request.

If you share reports publicly, review them first. File paths, session dates, or
session ids may reveal workflow details.

## Limitations

- Codex rollout log formats are not a public stability contract and may change.
- Estimated costs are not billing records.
- Model, effort, and mode are inferred from structured `turn_context` records.
  Older logs without those records will show `unknown`.
- Codex logs do not currently expose a reliable historical service tier per
  token event, so the tool does not claim whether a past event used `fast`. Standard is the default assumption; use
  `--pricing-tier` to choose another tier for the report.

## License

MIT
