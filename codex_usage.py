#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable


TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)

GPT55_INPUT_USD_PER_1M = 5.00
GPT55_CACHED_INPUT_USD_PER_1M = 0.50
GPT55_OUTPUT_USD_PER_1M = 30.00
PRICING_LABEL = (
    "Estimated with GPT-5.5 API text rates: "
    "$5.00/M uncached input, $0.50/M cached input, $30.00/M output"
)

SESSION_ID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens + other.reasoning_output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def delta_from(self, baseline: "Usage") -> "Usage":
        return Usage(
            input_tokens=max(0, self.input_tokens - baseline.input_tokens),
            cached_input_tokens=max(0, self.cached_input_tokens - baseline.cached_input_tokens),
            output_tokens=max(0, self.output_tokens - baseline.output_tokens),
            reasoning_output_tokens=max(
                0, self.reasoning_output_tokens - baseline.reasoning_output_tokens
            ),
            total_tokens=max(0, self.total_tokens - baseline.total_tokens),
        )

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)

    @classmethod
    def from_mapping(cls, value: dict | None) -> "Usage":
        value = value or {}
        return cls(**{key: int(value.get(key) or 0) for key in TOKEN_KEYS})

    def as_dict(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in TOKEN_KEYS}


@dataclass(frozen=True)
class TokenEvent:
    timestamp: datetime
    usage: Usage
    session_id: str
    path: Path


@dataclass(frozen=True)
class ReportRow:
    label: str
    usage: Usage
    sessions: int


@dataclass(frozen=True)
class Report:
    title: str
    start: datetime
    end: datetime
    group_by: str
    rows: list[ReportRow]
    totals: Usage
    sessions_counted: int
    files_counted: int
    events_counted: int


def default_roots(codex_home: Path | None = None) -> list[Path]:
    home = codex_home or Path.home() / ".codex"
    return [home / "sessions", home / "archived_sessions"]


def session_id_for_path(path: Path) -> str:
    match = SESSION_ID_RE.search(path.name)
    return match.group(1) if match else str(path)


def discover_rollout_files(roots: Iterable[Path]) -> list[Path]:
    by_session: dict[str, list[Path]] = {}
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            continue
        for path in root.rglob("rollout-*.jsonl"):
            by_session.setdefault(session_id_for_path(path), []).append(path)

    files: list[Path] = []
    for paths in by_session.values():
        paths.sort(
            key=lambda p: (
                1 if "archived_sessions" in p.parts else 0,
                -p.stat().st_mtime,
            )
        )
        files.append(paths[0])
    return sorted(files)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iter_token_events(path: Path) -> Iterable[TokenEvent]:
    session_id = session_id_for_path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"token_count"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            payload = record.get("payload") or {}
            if record.get("type") != "event_msg" or payload.get("type") != "token_count":
                continue

            info = payload.get("info") or {}
            usage = Usage.from_mapping(info.get("total_token_usage"))
            if usage.total_tokens == 0 and usage.input_tokens == 0 and usage.output_tokens == 0:
                continue

            try:
                timestamp = parse_timestamp(record["timestamp"])
            except (KeyError, ValueError):
                continue

            yield TokenEvent(
                timestamp=timestamp,
                usage=usage,
                session_id=session_id,
                path=path,
            )


def ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return value.astimezone(timezone.utc)


def local_midnight(day: date, tzinfo) -> datetime:
    return datetime.combine(day, time.min, tzinfo=tzinfo)


def parse_date(value: str, tzinfo) -> datetime:
    try:
        return local_midnight(date.fromisoformat(value), tzinfo)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def parse_duration(value: str) -> timedelta:
    match = re.fullmatch(r"\s*(\d+)\s*([hdwm])\s*", value.lower())
    if not match:
        raise argparse.ArgumentTypeError("duration must look like 24h, 7d, 2w, or 3m")
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    if unit == "w":
        return timedelta(weeks=amount)
    return timedelta(days=amount * 30)


def resolve_timeframe(args: argparse.Namespace, now: datetime | None = None) -> tuple[datetime, datetime, str]:
    local_tz = datetime.now().astimezone().tzinfo
    now = now or datetime.now(local_tz)
    today = now.date()

    if args.today:
        start = local_midnight(today, local_tz)
        end = now
        label = "Today"
    elif args.yesterday:
        start = local_midnight(today - timedelta(days=1), local_tz)
        end = local_midnight(today, local_tz)
        label = "Yesterday"
    elif args.week:
        start = local_midnight(today - timedelta(days=today.weekday()), local_tz)
        end = now
        label = "Current Week"
    elif args.last_week:
        this_week = local_midnight(today - timedelta(days=today.weekday()), local_tz)
        start = this_week - timedelta(days=7)
        end = this_week
        label = "Last Week"
    elif args.month:
        start = local_midnight(today.replace(day=1), local_tz)
        end = now
        label = "Current Month"
    elif args.since or args.until:
        start = parse_date(args.since, local_tz) if args.since else datetime.min.replace(tzinfo=timezone.utc)
        if args.until:
            # Treat --until as inclusive for dates by ending at next local midnight.
            end = parse_date(args.until, local_tz) + timedelta(days=1)
        else:
            end = now
        label = "Custom Range"
    else:
        duration = parse_duration(args.last)
        end = now
        start = now - duration
        label = f"Last {args.last}"

    start_utc = ensure_aware_utc(start)
    end_utc = ensure_aware_utc(end)
    if start_utc >= end_utc:
        raise SystemExit("error: start time must be before end time")
    return start_utc, end_utc, label


def group_label(timestamp: datetime, group_by: str, session_id: str, tzinfo) -> str:
    local = timestamp.astimezone(tzinfo)
    if group_by == "day":
        return local.date().isoformat()
    if group_by == "week":
        year, week, _ = local.isocalendar()
        return f"{year}-W{week:02d}"
    if group_by == "month":
        return f"{local.year:04d}-{local.month:02d}"
    if group_by == "session":
        return f"{local:%Y-%m-%d %H:%M} {session_id[:8]}"
    raise ValueError(f"unknown group: {group_by}")


def build_report(
    roots: Iterable[Path],
    start: datetime,
    end: datetime,
    group_by: str = "day",
    title: str | None = None,
    tzinfo=None,
) -> Report:
    start = ensure_aware_utc(start)
    end = ensure_aware_utc(end)
    tzinfo = tzinfo or datetime.now().astimezone().tzinfo

    files = discover_rollout_files(roots)
    grouped: dict[str, Usage] = {}
    grouped_sessions: dict[str, set[str]] = {}
    session_labels: dict[str, str] = {}
    sessions_with_usage: set[str] = set()
    files_counted = 0
    events_counted = 0

    for path in files:
        previous = Usage()
        saw_event = False
        file_had_window_usage = False

        for event in iter_token_events(path):
            saw_event = True
            if event.timestamp < start:
                previous = event.usage
                continue
            if event.timestamp >= end:
                break

            delta = event.usage.delta_from(previous)
            previous = event.usage
            if delta.total_tokens == 0:
                continue

            if group_by == "session":
                label = event.session_id
                session_labels.setdefault(
                    event.session_id,
                    group_label(event.timestamp, group_by, event.session_id, tzinfo),
                )
            else:
                label = group_label(event.timestamp, group_by, event.session_id, tzinfo)
            grouped[label] = grouped.get(label, Usage()) + delta
            grouped_sessions.setdefault(label, set()).add(event.session_id)
            sessions_with_usage.add(event.session_id)
            events_counted += 1
            file_had_window_usage = True

        if saw_event and file_had_window_usage:
            files_counted += 1

    rows = [
        ReportRow(
            label=session_labels.get(label, label),
            usage=usage,
            sessions=len(grouped_sessions.get(label, set())),
        )
        for label, usage in grouped.items()
    ]
    rows.sort(key=lambda row: row.label)
    if group_by == "session":
        rows.sort(key=lambda row: row.usage.total_tokens, reverse=True)

    totals = Usage()
    for row in rows:
        totals += row.usage

    report_title = title or f"Codex Token Usage Report - {group_by.title()}"
    return Report(
        title=report_title,
        start=start,
        end=end,
        group_by=group_by,
        rows=rows,
        totals=totals,
        sessions_counted=len(sessions_with_usage),
        files_counted=files_counted,
        events_counted=events_counted,
    )


def comma(value: int) -> str:
    return f"{value:,}"


def estimate_cost_usd(usage: Usage) -> float:
    return (
        usage.uncached_input_tokens * GPT55_INPUT_USD_PER_1M
        + usage.cached_input_tokens * GPT55_CACHED_INPUT_USD_PER_1M
        + usage.output_tokens * GPT55_OUTPUT_USD_PER_1M
    ) / 1_000_000


def dollars(value: float) -> str:
    return f"${value:,.2f}"


def ansi(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def visible_len(text: str) -> int:
    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def pad(text: str, width: int, align: str = "left") -> str:
    clean_width = visible_len(text)
    padding = max(0, width - clean_width)
    if align == "right":
        return " " * padding + text
    if align == "center":
        left = padding // 2
        return " " * left + text + " " * (padding - left)
    return text + " " * padding


def make_table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> str:
    widths = [
        max(visible_len(headers[i]), *(visible_len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def line(values: list[str]) -> str:
        cells = [
            " " + pad(value, widths[i], aligns[i]) + " "
            for i, value in enumerate(values)
        ]
        return "│" + "│".join(cells) + "│"

    output = [border("┌", "┬", "┐"), line(headers), border("├", "┼", "┤")]
    for index, row in enumerate(rows):
        output.append(line(row))
        if index != len(rows) - 1:
            output.append(border("├", "┼", "┤"))
    output.append(border("└", "┴", "┘"))
    return "\n".join(output)


def render_report(report: Report, color: bool = True, limit: int | None = None) -> str:
    rows = report.rows[:limit] if limit else report.rows
    cyan = "96"
    yellow = "93"
    muted = "2"

    headers = [
        ansi("Date" if report.group_by != "session" else "Session", cyan, color),
        ansi("Sessions", cyan, color),
        ansi("Input", cyan, color),
        ansi("Cached Input", cyan, color),
        ansi("Uncached", cyan, color),
        ansi("Output", cyan, color),
        ansi("Reasoning", cyan, color),
        ansi("Total Tokens", cyan, color),
        ansi("Cost (USD)", cyan, color),
    ]

    body: list[list[str]] = []
    for row in rows:
        body.append(
            [
                row.label,
                comma(row.sessions),
                comma(row.usage.input_tokens),
                comma(row.usage.cached_input_tokens),
                comma(row.usage.uncached_input_tokens),
                comma(row.usage.output_tokens),
                comma(row.usage.reasoning_output_tokens),
                comma(row.usage.total_tokens),
                dollars(estimate_cost_usd(row.usage)),
            ]
        )

    total_label = ansi("Total", yellow, color)
    total = report.totals
    body.append(
        [
            total_label,
            ansi(comma(report.sessions_counted), yellow, color),
            ansi(comma(total.input_tokens), yellow, color),
            ansi(comma(total.cached_input_tokens), yellow, color),
            ansi(comma(total.uncached_input_tokens), yellow, color),
            ansi(comma(total.output_tokens), yellow, color),
            ansi(comma(total.reasoning_output_tokens), yellow, color),
            ansi(comma(total.total_tokens), yellow, color),
            ansi(dollars(estimate_cost_usd(total)), yellow, color),
        ]
    )

    aligns = ["left", "right", "right", "right", "right", "right", "right", "right", "right"]
    table = make_table(headers, body, aligns)
    local_tz = datetime.now().astimezone().tzinfo
    subtitle = (
        f"{report.start.astimezone(local_tz):%Y-%m-%d %H:%M} -> "
        f"{report.end.astimezone(local_tz):%Y-%m-%d %H:%M %Z}"
    )
    meta = (
        f"{report.sessions_counted} sessions · {report.events_counted} token events · "
        f"{report.files_counted} rollout files"
    )
    cost_note = PRICING_LABEL + "; reasoning is included in output"
    box_width = max(
        visible_len(report.title),
        visible_len(subtitle),
        visible_len(meta),
        visible_len(cost_note),
    ) + 4
    title_box = "\n".join(
        [
            "┌" + "─" * box_width + "┐",
            "│ " + pad(report.title, box_width - 2, "center") + " │",
            "│ " + pad(ansi(subtitle, muted, color), box_width - 2, "center") + " │",
            "│ " + pad(ansi(meta, muted, color), box_width - 2, "center") + " │",
            "│ " + pad(ansi(cost_note, muted, color), box_width - 2, "center") + " │",
            "└" + "─" * box_width + "┘",
        ]
    )
    if not report.rows:
        return f"{title_box}\n\nNo token usage found for this timeframe."
    return f"{title_box}\n\n{table}"


def report_to_json(report: Report) -> dict:
    return {
        "title": report.title,
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
        "group_by": report.group_by,
        "sessions_counted": report.sessions_counted,
        "files_counted": report.files_counted,
        "events_counted": report.events_counted,
        "totals": {
            **report.totals.as_dict(),
            "uncached_input_tokens": report.totals.uncached_input_tokens,
            "estimated_cost_usd": estimate_cost_usd(report.totals),
        },
        "rows": [
            {
                "label": row.label,
                "sessions": row.sessions,
                **row.usage.as_dict(),
                "uncached_input_tokens": row.usage.uncached_input_tokens,
                "estimated_cost_usd": estimate_cost_usd(row.usage),
            }
            for row in report.rows
        ],
        "pricing": {
            "model": "GPT-5.5",
            "input_usd_per_1m": GPT55_INPUT_USD_PER_1M,
            "cached_input_usd_per_1m": GPT55_CACHED_INPUT_USD_PER_1M,
            "output_usd_per_1m": GPT55_OUTPUT_USD_PER_1M,
            "note": "Estimate only. Local ChatGPT-auth Codex usage is not API billing.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-usage",
        description="Show local Codex token usage from ~/.codex rollout logs.",
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--last", default="7d", help="rolling window: 24h, 7d, 2w, 3m (default: 7d)")
    window.add_argument("--today", action="store_true", help="show usage since local midnight")
    window.add_argument("--yesterday", action="store_true", help="show yesterday's usage")
    window.add_argument("--week", action="store_true", help="show current calendar week")
    window.add_argument("--last-week", action="store_true", help="show previous calendar week")
    window.add_argument("--month", action="store_true", help="show current calendar month")
    parser.add_argument("--since", help="start date, local time, YYYY-MM-DD")
    parser.add_argument("--until", help="end date, inclusive, local time, YYYY-MM-DD")
    parser.add_argument(
        "--group-by",
        choices=("day", "week", "month", "session"),
        default="day",
        help="aggregation level (default: day)",
    )
    parser.add_argument("--limit", type=int, help="limit displayed rows, useful with --group-by session")
    parser.add_argument("--codex-home", type=Path, default=Path.home() / ".codex")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="terminal color mode (default: auto)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.since or args.until) and args.last != "7d":
        parser.error("--since/--until cannot be combined with --last")

    start, end, timeframe_label = resolve_timeframe(args)
    title = f"Codex Token Usage Report - {args.group_by.title()} · {timeframe_label}"
    report = build_report(
        roots=default_roots(args.codex_home),
        start=start,
        end=end,
        group_by=args.group_by,
        title=title,
    )

    if args.json:
        print(json.dumps(report_to_json(report), indent=2))
        return 0

    color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
    print(render_report(report, color=color, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
