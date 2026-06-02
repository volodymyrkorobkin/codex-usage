#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import textwrap
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
UNKNOWN_METADATA = "unknown"

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
class TurnMetadata:
    model: str = UNKNOWN_METADATA
    effort: str = UNKNOWN_METADATA
    mode: str = UNKNOWN_METADATA

    @classmethod
    def from_payload(cls, payload: dict | None) -> "TurnMetadata":
        payload = payload or {}
        collaboration_mode = payload.get("collaboration_mode") or {}
        settings = collaboration_mode.get("settings") or {}

        return cls(
            model=metadata_value(payload.get("model") or settings.get("model")),
            effort=metadata_value(
                payload.get("effort") or settings.get("reasoning_effort")
            ),
            mode=metadata_value(collaboration_mode.get("mode")),
        )


@dataclass(frozen=True)
class TokenEvent:
    timestamp: datetime
    usage: Usage
    session_id: str
    path: Path
    metadata: TurnMetadata


@dataclass(frozen=True)
class ReportRow:
    label: str
    usage: Usage
    sessions: int
    models: tuple[str, ...] = ()
    efforts: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class TableColumn:
    key: str
    header: str
    align: str
    wrap: bool = False
    min_width: int | None = None


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


def metadata_value(value) -> str:
    if value is None:
        return UNKNOWN_METADATA
    value = str(value).strip()
    return value or UNKNOWN_METADATA


def iter_token_events(path: Path) -> Iterable[TokenEvent]:
    session_id = session_id_for_path(path)
    metadata = TurnMetadata()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"token_count"' not in line and '"turn_context"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            payload = record.get("payload") or {}
            if record.get("type") == "turn_context":
                metadata = TurnMetadata.from_payload(payload)
                continue

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
                metadata=metadata,
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
        raise argparse.ArgumentTypeError("duration must look like 24h, 7d, 30d, 2w, or 3m")
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


def group_label(
    timestamp: datetime,
    group_by: str,
    session_id: str,
    tzinfo,
    metadata: TurnMetadata,
) -> str:
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
    if group_by == "model":
        return metadata.model
    if group_by == "effort":
        return metadata.effort
    if group_by == "mode":
        return metadata.mode
    if group_by == "model-effort":
        return f"{metadata.model} · {metadata.effort}"
    raise ValueError(f"unknown group: {group_by}")


def label_header(group_by: str) -> str:
    return {
        "day": "Date",
        "week": "Week",
        "month": "Month",
        "session": "Session",
        "model": "Model",
        "effort": "Effort",
        "mode": "Mode",
        "model-effort": "Model / Effort",
    }.get(group_by, "Group")


def sorted_values(values: set[str]) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: (value == UNKNOWN_METADATA, value)))


def summarize_values(values: Iterable[str], limit: int = 5) -> str:
    unique = [value for value in sorted_values(set(values)) if value]
    if not unique:
        return UNKNOWN_METADATA
    if len(unique) <= limit:
        return ", ".join(unique)
    return ", ".join(unique[:limit]) + f", +{len(unique) - limit}"


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
    grouped_models: dict[str, set[str]] = {}
    grouped_efforts: dict[str, set[str]] = {}
    grouped_modes: dict[str, set[str]] = {}
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
                    group_label(
                        event.timestamp,
                        group_by,
                        event.session_id,
                        tzinfo,
                        event.metadata,
                    ),
                )
            else:
                label = group_label(
                    event.timestamp,
                    group_by,
                    event.session_id,
                    tzinfo,
                    event.metadata,
                )
            grouped[label] = grouped.get(label, Usage()) + delta
            grouped_sessions.setdefault(label, set()).add(event.session_id)
            grouped_models.setdefault(label, set()).add(event.metadata.model)
            grouped_efforts.setdefault(label, set()).add(event.metadata.effort)
            grouped_modes.setdefault(label, set()).add(event.metadata.mode)
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
            models=sorted_values(grouped_models.get(label, set())),
            efforts=sorted_values(grouped_efforts.get(label, set())),
            modes=sorted_values(grouped_modes.get(label, set())),
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


def detect_terminal_width() -> int:
    return shutil.get_terminal_size(fallback=(120, 24)).columns


def wrap_text(text: str, width: int, break_long_words: bool = True) -> list[str]:
    if width <= 0:
        return [text]
    lines: list[str] = []
    for part in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(
            part,
            width=width,
            break_long_words=break_long_words,
            break_on_hyphens=True,
        )
        lines.extend(wrapped or [""])
    return lines


def wrap_metadata_text(text: str, width: int) -> list[str]:
    parts = [part.strip() for part in str(text).split(", ")]
    if len(parts) == 1:
        return wrap_text(text, width)

    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}, {part}"
        if not current or visible_len(candidate) <= width:
            current = candidate
            continue
        lines.extend(wrap_text(current, width))
        current = part
    if current:
        lines.extend(wrap_text(current, width))
    return lines or [""]


def table_width(widths: list[int]) -> int:
    return sum(widths) + (3 * len(widths)) + 1


def fit_widths(
    widths: list[int],
    max_width: int | None,
    wrap_columns: set[int],
    min_widths: list[int],
) -> list[int]:
    if not max_width:
        return widths

    fitted = widths[:]
    while table_width(fitted) > max_width:
        candidates = [
            index
            for index in wrap_columns
            if fitted[index] > min_widths[index]
        ]
        if not candidates:
            break
        widest = max(candidates, key=lambda index: fitted[index])
        fitted[widest] -= 1
    return fitted


def make_table(
    headers: list[str],
    rows: list[list[str]],
    aligns: list[str],
    max_width: int | None = None,
    wrap_columns: set[int] | None = None,
    min_widths: list[int] | None = None,
    color: bool = True,
    header_style: str | None = None,
    total_style: str | None = None,
) -> str:
    wrap_columns = wrap_columns or set()
    widths = [
        max(visible_len(headers[i]), *(visible_len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    min_widths = min_widths or [visible_len(headers[i]) for i in range(len(headers))]
    widths = fit_widths(widths, max_width, wrap_columns, min_widths)

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def cell_lines(value: str, index: int) -> list[str]:
        if index not in wrap_columns:
            return str(value).splitlines() or [""]
        return wrap_metadata_text(str(value), widths[index])

    def line(values: list[str], style: str | None = None) -> list[str]:
        expanded = [cell_lines(value, index) for index, value in enumerate(values)]
        height = max(len(value_lines) for value_lines in expanded)
        output: list[str] = []
        for line_index in range(height):
            row_values = [
                value_lines[line_index] if line_index < len(value_lines) else ""
                for value_lines in expanded
            ]
            cells = [
                " " + pad(value, widths[i], aligns[i]) + " "
                for i, value in enumerate(row_values)
            ]
            if style:
                cells = [ansi(cell, style, color) for cell in cells]
            output.append("│" + "│".join(cells) + "│")
        return output

    output = [border("┌", "┬", "┐")]
    output.extend(line(headers, header_style))
    output.append(border("├", "┼", "┤"))
    for index, row in enumerate(rows):
        style = total_style if index == len(rows) - 1 else None
        output.extend(line(row, style))
        if index != len(rows) - 1:
            output.append(border("├", "┼", "┤"))
    output.append(border("└", "┴", "┘"))
    return "\n".join(output)


def table_columns(report: Report, table_mode: str) -> list[TableColumn]:
    sessions_header = "Sess" if table_mode == "compact" else "Sessions"
    common = [
        TableColumn("label", label_header(report.group_by), "left"),
        TableColumn("sessions", sessions_header, "right"),
        TableColumn("models", "Models", "left", wrap=True, min_width=8),
        TableColumn("efforts", "Efforts", "left", wrap=True, min_width=7),
        TableColumn("modes", "Modes", "left", wrap=True, min_width=7),
    ]
    if table_mode == "compact":
        return common + [
            TableColumn("total", "Total", "right"),
            TableColumn("cost", "Cost", "right"),
        ]
    return common + [
        TableColumn("input", "Input", "right"),
        TableColumn("cached_input", "Cached Input", "right"),
        TableColumn("uncached", "Uncached", "right"),
        TableColumn("output", "Output", "right"),
        TableColumn("reasoning", "Reasoning", "right"),
        TableColumn("total", "Total Tokens", "right"),
        TableColumn("cost", "Cost (USD)", "right"),
    ]


def value_for_column(key: str, row: ReportRow) -> str:
    values = {
        "label": row.label,
        "sessions": comma(row.sessions),
        "models": summarize_values(row.models),
        "efforts": summarize_values(row.efforts),
        "modes": summarize_values(row.modes),
        "input": comma(row.usage.input_tokens),
        "cached_input": comma(row.usage.cached_input_tokens),
        "uncached": comma(row.usage.uncached_input_tokens),
        "output": comma(row.usage.output_tokens),
        "reasoning": comma(row.usage.reasoning_output_tokens),
        "total": comma(row.usage.total_tokens),
        "cost": dollars(estimate_cost_usd(row.usage)),
    }
    return values[key]


def total_row_for_columns(
    columns: list[TableColumn],
    rows: list[ReportRow],
    report: Report,
) -> list[str]:
    all_models = [model for row in rows for model in row.models]
    all_efforts = [effort for row in rows for effort in row.efforts]
    all_modes = [mode for row in rows for mode in row.modes]
    total = report.totals
    values = {
        "label": "Total",
        "sessions": comma(report.sessions_counted),
        "models": summarize_values(all_models),
        "efforts": summarize_values(all_efforts),
        "modes": summarize_values(all_modes),
        "input": comma(total.input_tokens),
        "cached_input": comma(total.cached_input_tokens),
        "uncached": comma(total.uncached_input_tokens),
        "output": comma(total.output_tokens),
        "reasoning": comma(total.reasoning_output_tokens),
        "total": comma(total.total_tokens),
        "cost": dollars(estimate_cost_usd(total)),
    }
    return [values[column.key] for column in columns]


def render_table_for_mode(
    report: Report,
    rows: list[ReportRow],
    table_mode: str,
    color: bool,
    max_width: int | None,
) -> str:
    columns = table_columns(report, table_mode)
    headers = [column.header for column in columns]
    body = [
        [value_for_column(column.key, row) for column in columns]
        for row in rows
    ]
    body.append(total_row_for_columns(columns, rows, report))
    aligns = [column.align for column in columns]
    wrap_columns = {index for index, column in enumerate(columns) if column.wrap}
    min_widths = [
        column.min_width or visible_len(column.header)
        for column in columns
    ]
    return make_table(
        headers,
        body,
        aligns,
        max_width=max_width,
        wrap_columns=wrap_columns,
        min_widths=min_widths,
        color=color,
        header_style="96",
        total_style="93",
    )


def render_usage_table(
    report: Report,
    rows: list[ReportRow],
    table_mode: str,
    color: bool,
    max_width: int,
) -> str:
    if table_mode == "auto":
        full_table = render_table_for_mode(report, rows, "full", color, max_width)
        if all(visible_len(line) <= max_width for line in full_table.splitlines()):
            return full_table
        return render_table_for_mode(report, rows, "compact", color, max_width)
    return render_table_for_mode(report, rows, table_mode, color, max_width)


def make_title_box(
    title: str,
    subtitle: str,
    meta: str,
    cost_note: str,
    color: bool,
    max_width: int,
) -> str:
    items = [
        (title, None),
        (subtitle, "2"),
        (meta, "2"),
        (cost_note, "2"),
    ]
    content_width = max(
        20,
        min(max(visible_len(text) for text, _ in items), max_width - 4),
    )
    wrapped: list[tuple[str, str | None]] = []
    for text, style in items:
        wrapped.extend((line, style) for line in wrap_text(text, content_width))
    box_width = max(visible_len(line) for line, _ in wrapped) + 4

    output = ["┌" + "─" * box_width + "┐"]
    for text, style in wrapped:
        value = ansi(text, style, color) if style else text
        output.append("│ " + pad(value, box_width - 2, "center") + " │")
    output.append("└" + "─" * box_width + "┘")
    return "\n".join(output)


def render_report(
    report: Report,
    color: bool = True,
    limit: int | None = None,
    terminal_width: int | None = None,
    table_mode: str = "auto",
) -> str:
    rows = report.rows[:limit] if limit else report.rows
    width = terminal_width or detect_terminal_width()
    table = render_usage_table(report, rows, table_mode, color, width)
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
    title_box = make_title_box(
        report.title,
        subtitle,
        meta,
        cost_note,
        color,
        width,
    )
    if not report.rows:
        return f"{title_box}\n\nNo token usage found for this timeframe."
    return f"{title_box}\n\n{table}"


def report_to_json(report: Report) -> dict:
    all_models = [model for row in report.rows for model in row.models]
    all_efforts = [effort for row in report.rows for effort in row.efforts]
    all_modes = [mode for row in report.rows for mode in row.modes]
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
            "models": list(sorted_values(set(all_models))),
            "efforts": list(sorted_values(set(all_efforts))),
            "modes": list(sorted_values(set(all_modes))),
        },
        "rows": [
            {
                "label": row.label,
                "sessions": row.sessions,
                "models": list(row.models),
                "efforts": list(row.efforts),
                "modes": list(row.modes),
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
    window.add_argument("--last", default="30d", help="rolling window: 24h, 7d, 30d, 2w, 3m (default: 30d)")
    window.add_argument("--today", action="store_true", help="show usage since local midnight")
    window.add_argument("--yesterday", action="store_true", help="show yesterday's usage")
    window.add_argument("--week", action="store_true", help="show current calendar week")
    window.add_argument("--last-week", action="store_true", help="show previous calendar week")
    window.add_argument("--month", action="store_true", help="show current calendar month")
    parser.add_argument("--since", help="start date, local time, YYYY-MM-DD")
    parser.add_argument("--until", help="end date, inclusive, local time, YYYY-MM-DD")
    parser.add_argument(
        "--group-by",
        choices=("day", "week", "month", "session", "model", "effort", "mode", "model-effort"),
        default="day",
        help="aggregation level (default: day)",
    )
    parser.add_argument("--limit", type=int, help="limit displayed rows, useful with --group-by session")
    parser.add_argument("--codex-home", type=Path, default=Path.home() / ".codex")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--table",
        choices=("auto", "full", "compact"),
        default="auto",
        help="table layout mode (default: auto)",
    )
    parser.add_argument("--width", type=int, help="override detected terminal width")
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

    if (args.since or args.until) and args.last != "30d":
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
    print(
        render_report(
            report,
            color=color,
            limit=args.limit,
            terminal_width=args.width,
            table_mode=args.table,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
