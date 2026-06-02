from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import codex_usage


def token_event(timestamp: str, total: dict[str, int], last: dict[str, int] | None = None) -> str:
    payload = {
        "timestamp": timestamp,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": total,
                "last_token_usage": last or total,
                "model_context_window": 258400,
            },
            "rate_limits": {"plan_type": "business"},
        },
    }
    return json.dumps(payload)


def turn_context(
    timestamp: str,
    model: str = "gpt-5.5",
    effort: str = "xhigh",
    mode: str = "default",
) -> str:
    payload = {
        "timestamp": timestamp,
        "type": "turn_context",
        "payload": {
            "model": model,
            "effort": effort,
            "collaboration_mode": {
                "mode": mode,
                "settings": {
                    "model": model,
                    "reasoning_effort": effort,
                },
            },
        },
    }
    return json.dumps(payload)


class CodexUsageTests(unittest.TestCase):
    def test_default_window_is_last_30_days(self) -> None:
        parser = codex_usage.build_parser()
        args = parser.parse_args([])

        start, end, label = codex_usage.resolve_timeframe(
            args,
            now=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(args.last, "30d")
        self.assertEqual(label, "Last 30d")
        self.assertEqual(end - start, codex_usage.parse_duration("30d"))

    def test_metadata_wrapping_does_not_start_lines_with_commas(self) -> None:
        self.assertEqual(
            codex_usage.wrap_metadata_text("default, plan, unknown", 7),
            ["default", "plan", "unknown"],
        )

    def test_rollup_uses_cumulative_delta_within_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rollout-2026-06-01T00-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            log.write_text(
                "\n".join(
                    [
                        token_event(
                            "2026-05-24T23:00:00Z",
                            {
                                "input_tokens": 100,
                                "cached_input_tokens": 40,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 3,
                                "total_tokens": 110,
                            },
                        ),
                        token_event(
                            "2026-05-25T12:00:00Z",
                            {
                                "input_tokens": 250,
                                "cached_input_tokens": 160,
                                "output_tokens": 30,
                                "reasoning_output_tokens": 8,
                                "total_tokens": 280,
                            },
                            {
                                "input_tokens": 150,
                                "cached_input_tokens": 120,
                                "output_tokens": 20,
                                "reasoning_output_tokens": 5,
                                "total_tokens": 170,
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            report = codex_usage.build_report(
                roots=[Path(tmp)],
                start=datetime(2026, 5, 25, 0, 0, tzinfo=timezone.utc),
                end=datetime(2026, 5, 26, 0, 0, tzinfo=timezone.utc),
                group_by="day",
            )

        self.assertEqual(report.totals.total_tokens, 170)
        self.assertEqual(report.totals.input_tokens, 150)
        self.assertEqual(report.totals.cached_input_tokens, 120)
        self.assertEqual(report.totals.output_tokens, 20)
        self.assertEqual(report.totals.reasoning_output_tokens, 5)
        self.assertEqual(len(report.rows), 1)
        self.assertEqual(report.rows[0].label, "2026-05-25")

    def test_deduplicates_active_and_archived_copy_by_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            active = Path(tmp) / "sessions"
            archived = Path(tmp) / "archived_sessions"
            active.mkdir()
            archived.mkdir()
            name = "rollout-2026-06-01T00-00-00-11111111-2222-3333-4444-555555555555.jsonl"
            content = token_event(
                "2026-06-01T01:00:00Z",
                {
                    "input_tokens": 1000,
                    "cached_input_tokens": 800,
                    "output_tokens": 50,
                    "reasoning_output_tokens": 10,
                    "total_tokens": 1050,
                },
            )
            (active / name).write_text(content, encoding="utf-8")
            (archived / name).write_text(content, encoding="utf-8")

            report = codex_usage.build_report(
                roots=[active, archived],
                start=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
                group_by="day",
            )

        self.assertEqual(report.totals.total_tokens, 1050)
        self.assertEqual(report.sessions_counted, 1)

    def test_session_grouping_keeps_one_row_per_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rollout-2026-06-01T00-00-00-99999999-2222-3333-4444-555555555555.jsonl"
            log.write_text(
                "\n".join(
                    [
                        token_event(
                            "2026-06-01T01:00:00Z",
                            {
                                "input_tokens": 100,
                                "cached_input_tokens": 80,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 2,
                                "total_tokens": 110,
                            },
                        ),
                        token_event(
                            "2026-06-01T01:01:00Z",
                            {
                                "input_tokens": 250,
                                "cached_input_tokens": 180,
                                "output_tokens": 20,
                                "reasoning_output_tokens": 5,
                                "total_tokens": 270,
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            report = codex_usage.build_report(
                roots=[Path(tmp)],
                start=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
                group_by="session",
            )

        self.assertEqual(len(report.rows), 1)
        self.assertEqual(report.rows[0].sessions, 1)
        self.assertEqual(report.rows[0].usage.total_tokens, 270)

    def test_token_events_inherit_latest_turn_context_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rollout-2026-06-01T00-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            log.write_text(
                "\n".join(
                    [
                        turn_context(
                            "2026-06-01T00:01:00Z",
                            model="gpt-5.4",
                            effort="high",
                            mode="plan",
                        ),
                        token_event(
                            "2026-06-01T00:02:00Z",
                            {
                                "input_tokens": 80,
                                "cached_input_tokens": 20,
                                "output_tokens": 20,
                                "reasoning_output_tokens": 5,
                                "total_tokens": 100,
                            },
                        ),
                        turn_context(
                            "2026-06-01T00:03:00Z",
                            model="gpt-5.5",
                            effort="xhigh",
                            mode="default",
                        ),
                        token_event(
                            "2026-06-01T00:04:00Z",
                            {
                                "input_tokens": 230,
                                "cached_input_tokens": 120,
                                "output_tokens": 70,
                                "reasoning_output_tokens": 25,
                                "total_tokens": 300,
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            report = codex_usage.build_report(
                roots=[Path(tmp)],
                start=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
                group_by="day",
            )

        self.assertEqual(len(report.rows), 1)
        self.assertEqual(report.rows[0].usage.total_tokens, 300)
        self.assertEqual(report.rows[0].models, ("gpt-5.4", "gpt-5.5"))
        self.assertEqual(report.rows[0].efforts, ("high", "xhigh"))
        self.assertEqual(report.rows[0].modes, ("default", "plan"))

    def test_group_by_model_effort_uses_context_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rollout-2026-06-01T00-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            log.write_text(
                "\n".join(
                    [
                        turn_context("2026-06-01T00:01:00Z", model="gpt-5.4", effort="high"),
                        token_event(
                            "2026-06-01T00:02:00Z",
                            {
                                "input_tokens": 100,
                                "cached_input_tokens": 20,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 3,
                                "total_tokens": 110,
                            },
                        ),
                        turn_context("2026-06-01T00:03:00Z", model="gpt-5.5", effort="xhigh"),
                        token_event(
                            "2026-06-01T00:04:00Z",
                            {
                                "input_tokens": 300,
                                "cached_input_tokens": 100,
                                "output_tokens": 30,
                                "reasoning_output_tokens": 8,
                                "total_tokens": 330,
                            },
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            report = codex_usage.build_report(
                roots=[Path(tmp)],
                start=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
                group_by="model-effort",
            )

        rows = {row.label: row for row in report.rows}
        self.assertEqual(rows["gpt-5.4 · high"].usage.total_tokens, 110)
        self.assertEqual(rows["gpt-5.5 · xhigh"].usage.total_tokens, 220)

    def test_render_table_includes_title_header_and_total_row(self) -> None:
        usage = codex_usage.Usage(
            input_tokens=1200,
            cached_input_tokens=900,
            output_tokens=40,
            reasoning_output_tokens=9,
            total_tokens=1240,
        )
        report = codex_usage.Report(
            title="Codex Token Usage Report - Daily",
            start=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
            group_by="day",
            rows=[
                codex_usage.ReportRow(
                    label="2026-06-01",
                    usage=usage,
                    sessions=1,
                    models=("gpt-5.5",),
                    efforts=("xhigh",),
                    modes=("default",),
                )
            ],
            totals=usage,
            sessions_counted=1,
            files_counted=1,
            events_counted=1,
        )

        rendered = codex_usage.render_report(
            report,
            color=False,
            terminal_width=180,
            table_mode="full",
        )

        self.assertIn("Codex Token Usage Report - Daily", rendered)
        self.assertIn("Date", rendered)
        self.assertIn("Models", rendered)
        self.assertIn("Efforts", rendered)
        self.assertIn("Modes", rendered)
        self.assertIn("gpt-5.5", rendered)
        self.assertIn("Cached Input", rendered)
        self.assertIn("1,240", rendered)
        self.assertIn("Total", rendered)

    def test_estimated_cost_uses_gpt55_cached_and_uncached_rates(self) -> None:
        usage = codex_usage.Usage(
            input_tokens=2_000_000,
            cached_input_tokens=1_500_000,
            output_tokens=100_000,
            reasoning_output_tokens=50_000,
            total_tokens=2_100_000,
        )

        cost = codex_usage.estimate_cost_usd(usage)

        self.assertAlmostEqual(cost, 6.25)

    def test_render_table_includes_estimated_cost_column(self) -> None:
        usage = codex_usage.Usage(
            input_tokens=1_000_000,
            cached_input_tokens=0,
            output_tokens=1_000_000,
            reasoning_output_tokens=500_000,
            total_tokens=2_000_000,
        )
        report = codex_usage.Report(
            title="Codex Token Usage Report - Daily",
            start=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
            group_by="day",
            rows=[
                codex_usage.ReportRow(
                    label="2026-06-01",
                    usage=usage,
                    sessions=1,
                    models=("gpt-5.5",),
                    efforts=("xhigh",),
                    modes=("default",),
                )
            ],
            totals=usage,
            sessions_counted=1,
            files_counted=1,
            events_counted=1,
        )

        rendered = codex_usage.render_report(
            report,
            color=False,
            terminal_width=180,
            table_mode="full",
        )

        self.assertIn("Cost (USD)", rendered)
        self.assertIn("$35.00", rendered)

    def test_auto_table_fits_requested_terminal_width(self) -> None:
        usage = codex_usage.Usage(
            input_tokens=1_765_756_172,
            cached_input_tokens=1_679_256_448,
            output_tokens=6_457_884,
            reasoning_output_tokens=2_163_395,
            total_tokens=1_773_627_014,
        )
        report = codex_usage.Report(
            title="Codex Token Usage Report - Daily",
            start=datetime(2026, 5, 3, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 6, 3, 0, 0, tzinfo=timezone.utc),
            group_by="day",
            rows=[
                codex_usage.ReportRow(
                    label="2026-06-02",
                    usage=usage,
                    sessions=185,
                    models=("codex-auto-review", "gpt-5.5", "unknown"),
                    efforts=("high", "low", "medium", "xhigh", "unknown"),
                    modes=("default", "plan", "unknown"),
                )
            ],
            totals=usage,
            sessions_counted=185,
            files_counted=1,
            events_counted=1,
        )

        rendered = codex_usage.render_report(
            report,
            color=False,
            terminal_width=80,
            table_mode="auto",
        )

        self.assertTrue(
            all(codex_usage.visible_len(line) <= 80 for line in rendered.splitlines()),
            rendered,
        )
        self.assertIn("codex-", rendered)
        self.assertIn("xhigh", rendered)
        self.assertIn("Cost", rendered)

    def test_full_table_can_be_forced_for_wide_output(self) -> None:
        usage = codex_usage.Usage(
            input_tokens=1_000_000,
            cached_input_tokens=500_000,
            output_tokens=25_000,
            reasoning_output_tokens=10_000,
            total_tokens=1_025_000,
        )
        report = codex_usage.Report(
            title="Codex Token Usage Report - Daily",
            start=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc),
            group_by="day",
            rows=[
                codex_usage.ReportRow(
                    label="2026-06-01",
                    usage=usage,
                    sessions=1,
                    models=("gpt-5.5",),
                    efforts=("xhigh",),
                    modes=("default",),
                )
            ],
            totals=usage,
            sessions_counted=1,
            files_counted=1,
            events_counted=1,
        )

        rendered = codex_usage.render_report(
            report,
            color=False,
            terminal_width=180,
            table_mode="full",
        )

        self.assertIn("Cached Input", rendered)
        self.assertIn("Uncached", rendered)
        self.assertIn("Reasoning", rendered)


if __name__ == "__main__":
    unittest.main()
