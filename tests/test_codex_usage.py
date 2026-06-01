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


class CodexUsageTests(unittest.TestCase):
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
            rows=[codex_usage.ReportRow(label="2026-06-01", usage=usage, sessions=1)],
            totals=usage,
            sessions_counted=1,
            files_counted=1,
            events_counted=1,
        )

        rendered = codex_usage.render_report(report, color=False)

        self.assertIn("Codex Token Usage Report - Daily", rendered)
        self.assertIn("Date", rendered)
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
            rows=[codex_usage.ReportRow(label="2026-06-01", usage=usage, sessions=1)],
            totals=usage,
            sessions_counted=1,
            files_counted=1,
            events_counted=1,
        )

        rendered = codex_usage.render_report(report, color=False)

        self.assertIn("Cost (USD)", rendered)
        self.assertIn("$35.00", rendered)


if __name__ == "__main__":
    unittest.main()
