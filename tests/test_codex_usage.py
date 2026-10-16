from __future__ import annotations

import json
import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import codex_usage
import codex_pricing
from unittest.mock import patch
from datetime import timedelta
from dataclasses import replace


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
        self.assertEqual(report.rows[0].label, "2026-06-01 01:00 99999999")

    def test_session_names_preserve_time_and_distinct_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            active = home / "sessions" / "2026" / "06"
            archived = home / "archived_sessions"
            active.mkdir(parents=True)
            archived.mkdir()
            ids = ["aaaaaaaa-2222-3333-4444-555555555555",
                   "bbbbbbbb-2222-3333-4444-555555555555"]
            for root, session_id in zip((active, archived), ids):
                (root / f"rollout-{session_id}.jsonl").write_text(
                    token_event("2026-06-01T01:00:00Z", {"total_tokens": 100}),
                    encoding="utf-8",
                )
            name = "Investigate, then repair the session usage report"
            (home / "session_index.jsonl").write_text(
                "\n".join([
                    json.dumps({"id": ids[0], "thread_name": "Old name"}),
                    "malformed record",
                    "null",
                    json.dumps({"id": ids[0], "thread_name": None}),
                    *(json.dumps({"id": sid, "thread_name": name}) for sid in ids),
                ]), encoding="utf-8",
            )
            report = codex_usage.build_report(
                roots=iter([home / "sessions", archived]),
                start=datetime(2026, 6, 1, tzinfo=timezone.utc),
                end=datetime(2026, 6, 2, tzinfo=timezone.utc),
                group_by="session",
                tzinfo=timezone.utc,
            )

        self.assertEqual(len(report.rows), 2)
        self.assertEqual(report.totals.total_tokens, 200)
        self.assertEqual(
            {row.label for row in report.rows},
            {f"2026-06-01 01:00 {sid[:8]} · {name}" for sid in ids},
        )
        exported = codex_usage.report_to_json(report)
        self.assertTrue(all(name in row["label"] for row in exported["rows"]))
        rendered = codex_usage.render_usage_table(
            report, report.rows, "auto", False, 80,
        )
        self.assertTrue(all(len(line) <= 80 for line in rendered.splitlines()), rendered)
        self.assertIn("01:00", rendered)
        self.assertIn("Investigate,", rendered)

    def test_session_index_normalizes_names_and_ignores_blank_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "session_index.jsonl").write_text(
                "\n".join(json.dumps(record) for record in [
                    {"id": "example", "thread_name": " Fix\n\tusage\x07 reports "},
                    {"id": "example", "thread_name": "   "},
                    {"id": 123, "thread_name": "Invalid id"},
                ]), encoding="utf-8",
            )
            self.assertEqual(codex_usage.load_session_titles([root]),
                             {"example": "Fix usage reports"})

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

    def test_estimated_cost_uses_actual_model_rates(self) -> None:
        usage = codex_usage.Usage(
            input_tokens=2_000_000,
            cached_input_tokens=1_500_000,
            output_tokens=100_000,
            reasoning_output_tokens=50_000,
            total_tokens=2_100_000,
        )

        pricing = codex_pricing.Pricing(codex_pricing.bundled_catalog())
        stamp = codex_pricing.utc(codex_pricing.BUNDLED_CHECKED_AT)
        self.assertAlmostEqual(pricing.cost(usage, "gpt-6-sol", stamp, 1000), 2.3)
        self.assertAlmostEqual(pricing.cost(usage, "gpt-6-luna", stamp, 1000), 0.115)
        self.assertAlmostEqual(pricing.cost(usage, "gpt-6.1-sol", stamp, 1000), 2.15)

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
        self.assertIn("N/A", rendered)

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


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.stamp = datetime(2026, 10, 1, tzinfo=timezone.utc)
        rates = dict(input=2, cached_input=0.2, cache_write=2.5, output=10,
                     long_input=4, long_cached_input=0.4, long_cache_write=5, long_output=15)
        self.data = dict(schema_version=1, checked_at=self.stamp.isoformat(), prices=[
            dict(model="gpt-6-sol", tier="standard", start=self.stamp.isoformat(),
                 end=None, date_basis="effective", source=codex_pricing.SOURCE_URL,
                 long_context_threshold=272000, rates=rates)])

    def test_changes_preserve_history_and_exclusive_end(self):
        changed = self.stamp + timedelta(days=1)
        rates = self.data['prices'][0]['rates'].copy()
        rates['input'] = 3
        data = codex_pricing.update_catalog(self.data, {('standard', 'gpt-6-sol'): rates}, changed)
        pricing = codex_pricing.Pricing(data)
        usage = codex_usage.Usage(input_tokens=1_000_000)
        self.assertEqual(pricing.cost(usage, 'gpt-6-sol', changed - timedelta(seconds=1), 100), 2)
        self.assertEqual(pricing.cost(usage, 'gpt-6-sol', changed, 100), 3)
        self.assertEqual(data['prices'][0]['end'], data['prices'][1]['start'])
        same = codex_pricing.update_catalog(data, {('standard', 'gpt-6-sol'): rates}, changed + timedelta(days=1))
        self.assertEqual(len(same['prices']), 2)
        self.assertIsNone(self.data['prices'][0]['end'])

    def test_unknown_and_historical_gaps_are_not_zero(self):
        pricing = codex_pricing.Pricing(self.data, strict_history=True)
        usage = codex_usage.Usage(input_tokens=10)
        self.assertAlmostEqual(pricing.cost(usage, 'unknown', self.stamp, 10), .00002)
        self.assertIsNone(pricing.cost(usage, 'unknown', self.stamp - timedelta(days=1), 10))
        self.assertIsNone(pricing.cost(usage, 'gpt-6-sol', self.stamp - timedelta(days=1), 10))
        pricing = codex_pricing.Pricing(self.data)
        self.assertIsNotNone(pricing.cost(usage, 'gpt-6-sol', self.stamp - timedelta(days=1), 10))
        self.assertIn('Current published rates used before recorded history', pricing.notes)

    def test_long_context_threshold_and_reasoning_not_double_counted(self):
        pricing = codex_pricing.Pricing(self.data)
        usage = codex_usage.Usage(input_tokens=1_000_000, cached_input_tokens=500_000,
                                  output_tokens=100_000, reasoning_output_tokens=50_000)
        self.assertAlmostEqual(pricing.cost(usage, 'gpt-6-sol', self.stamp, 272000), 2.1)
        self.assertAlmostEqual(pricing.cost(usage, 'gpt-6-sol', self.stamp, 272001), 3.7)

    def test_report_prices_mixed_models_before_grouping_and_keeps_full_total_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'rollout-fixture.jsonl'
            log.write_text('\n'.join([
                turn_context('2026-10-01T01:00:00Z', 'gpt-6-sol'),
                token_event('2026-10-01T01:01:00Z', dict(input_tokens=1000000, output_tokens=0, total_tokens=1000000)),
                turn_context('2026-10-01T02:00:00Z', 'gpt-6-luna'),
                token_event('2026-10-01T02:01:00Z', dict(input_tokens=2000000, output_tokens=0, total_tokens=2000000),
                            dict(input_tokens=1000000, output_tokens=0, total_tokens=1000000)),
            ]))
            data = json.loads(json.dumps(self.data))
            luna = json.loads(json.dumps(data['prices'][0]))
            luna['model'] = 'gpt-6-luna'
            luna['rates']['long_input'] = 0.2
            data['prices'].append(luna)
            daily = codex_usage.build_report([Path(tmp)], self.stamp, self.stamp + timedelta(days=1),
                                              pricing=codex_pricing.Pricing(data))
            models = codex_usage.build_report([Path(tmp)], self.stamp, self.stamp + timedelta(days=1),
                                               group_by='model', pricing=codex_pricing.Pricing(data))
        self.assertAlmostEqual(daily.estimated_cost_usd, 4.2)
        self.assertAlmostEqual(daily.rows[0].estimated_cost_usd, 4.2)
        self.assertAlmostEqual(models.estimated_cost_usd, 4.2)
        self.assertIn('$4.20', codex_usage.render_report(models, limit=1, color=False))
        self.assertEqual(codex_usage.report_to_json(daily)['totals']['estimated_cost_usd'], 4.2)

    def test_missing_model_uses_cheapest_available_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'rollout-fixture.jsonl'
            log.write_text(token_event('2026-10-01T01:00:00Z', dict(input_tokens=10, total_tokens=10)))
            report = codex_usage.build_report([Path(tmp)], self.stamp, self.stamp + timedelta(days=1),
                                              pricing=codex_pricing.Pricing(self.data))
        self.assertEqual(report.unpriced_events, 0)
        self.assertAlmostEqual(report.estimated_cost_usd, .00002)
        self.assertAlmostEqual(codex_usage.report_to_json(report)['rows'][0]['estimated_cost_usd'], .00002)
        self.assertEqual(report.pricing['fallback_models'][0]['priced_as'], 'gpt-6-sol')

    def test_auto_review_does_not_hide_priced_daily_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'rollout-review-fixture.jsonl'
            log.write_text('\n'.join([
                turn_context('2026-10-01T01:00:00Z', 'gpt-6-sol'),
                token_event('2026-10-01T01:01:00Z', dict(input_tokens=100000, total_tokens=100000)),
                turn_context('2026-10-01T02:00:00Z', 'codex-auto-review', effort='low'),
                token_event('2026-10-01T02:01:00Z', dict(input_tokens=101000, total_tokens=101000),
                            dict(input_tokens=1000, total_tokens=1000)),
            ]))
            reports = [codex_usage.build_report([Path(tmp)], self.stamp, self.stamp + timedelta(days=1),
                                               group_by=group, pricing=codex_pricing.Pricing(self.data))
                       for group in ('day', 'model')]
        daily, models = reports
        self.assertEqual(daily.totals.input_tokens, 101000)
        self.assertEqual(daily.unpriced_events, 0)
        self.assertEqual(daily.priced_events, 2)
        self.assertAlmostEqual(daily.estimated_cost_usd, 0.202)
        self.assertAlmostEqual(daily.known_cost_usd, 0.202)
        self.assertEqual(codex_usage.display_cost(daily.rows[0]), '$0.20')
        for mode in ('full', 'compact', 'auto'):
            rendered = codex_usage.render_report(daily, color=False, table_mode=mode)
            self.assertIn('$0.20', rendered)
            self.assertIn('Estimated codex-auto-review using cheapest available Codex model: gpt-6-sol', rendered)
        rows = {row.label: row for row in models.rows}
        self.assertEqual(codex_usage.display_cost(rows['codex-auto-review']), '$0.00')
        self.assertEqual(codex_usage.display_cost(rows['gpt-6-sol']), '$0.20')
        self.assertIn('$0.20', codex_usage.render_report(models, color=False, limit=1))
        data = codex_usage.report_to_json(daily)
        self.assertAlmostEqual(data['rows'][0]['known_cost_usd'], 0.202)
        self.assertAlmostEqual(data['totals']['known_cost_usd'], 0.202)
        self.assertEqual(data['totals']['priced_events'], 2)
        self.assertAlmostEqual(data['totals']['estimated_cost_usd'], 0.202)
        self.assertEqual(data['pricing']['fallback_models'][0]['model'], 'codex-auto-review')

    def test_unknown_model_uses_luna_and_discloses_proxy(self):
        pricing = codex_pricing.Pricing(codex_pricing.bundled_catalog())
        usage = codex_usage.Usage(input_tokens=1_000_000, cached_input_tokens=800_000,
                                 output_tokens=10_000, reasoning_output_tokens=9000)
        stamp = codex_pricing.utc(codex_pricing.BUNDLED_CHECKED_AT)
        self.assertAlmostEqual(pricing.cost(usage, 'codex-auto-review', stamp, 1000), .033)
        self.assertAlmostEqual(pricing.cost(usage, 'codex-auto-review', stamp, 300000), .0635)
        fallback = pricing.as_dict()['fallback_models'][0]
        self.assertEqual(fallback['priced_as'], 'gpt-6-luna')
        self.assertEqual(fallback['events'], 2)
        self.assertAlmostEqual(fallback['estimated_cost_usd'], .0965)

    def test_cheapest_unknown_proxy_changes_with_prices_and_token_mix(self):
        data = json.loads(json.dumps(self.data))
        other = json.loads(json.dumps(data['prices'][0]))
        other['model'] = 'gpt-6-terra'
        other['rates'].update(input=5, output=1)
        data['prices'].append(other)
        pricing = codex_pricing.Pricing(data)
        self.assertEqual(pricing.cost(codex_usage.Usage(output_tokens=1_000_000), 'new-model', self.stamp, 100), 1)
        self.assertEqual(pricing.cost(codex_usage.Usage(input_tokens=1_000_000), 'new-model', self.stamp, 100), 2)
        changed = self.stamp + timedelta(days=1)
        rates = other['rates'].copy()
        rates['output'] = 20
        updated = codex_pricing.Pricing(codex_pricing.update_catalog(data, {('standard', other['model']): rates}, changed))
        self.assertEqual(updated.cost(codex_usage.Usage(output_tokens=1_000_000), 'new-model', changed, 100), 10)

    def test_unknown_proxy_respects_tier_context_and_history(self):
        usage = codex_usage.Usage(input_tokens=1_000_000)
        pricing = codex_pricing.Pricing(self.data, tier='fast')
        self.assertIsNone(pricing.cost(usage, 'new-model', self.stamp, 100))
        self.assertIsNone(pricing.cost(usage, 'gpt-6-sol', self.stamp, 100))
        data = json.loads(json.dumps(self.data))
        data['prices'][0]['rates']['cached_input'] = None
        pricing = codex_pricing.Pricing(data)
        self.assertIsNone(pricing.cost(codex_usage.Usage(input_tokens=100, cached_input_tokens=100), 'new-model', self.stamp, 100))
        pricing = codex_pricing.Pricing(self.data)
        self.assertEqual(pricing.cost(usage, 'new-model', self.stamp - timedelta(days=1), 100), 2)
        self.assertIn('Current published rates used before recorded history', pricing.notes)

    def test_zero_priced_subtotal_is_distinct_from_no_priced_events(self):
        row = codex_usage.ReportRow('mixed', codex_usage.Usage(), 1,
                                    priced_events=1, unpriced_events=1)
        self.assertEqual(codex_usage.display_cost(row), '$0.00*')
        unknown = codex_usage.ReportRow('unknown', codex_usage.Usage(), 1, unpriced_events=1)
        self.assertEqual(codex_usage.display_cost(unknown), 'N/A')

    def test_catalog_validation_rejects_overlaps_invalid_rates_and_dates(self):
        for mutation in ('overlap', 'negative', 'naive'):
            data = json.loads(json.dumps(self.data))
            if mutation == 'overlap':
                data['prices'].append(data['prices'][0].copy())
            elif mutation == 'negative':
                data['prices'][0]['rates']['input'] = -1
            else:
                data['prices'][0]['start'] = '2026-10-01'
            with self.assertRaises(ValueError):
                codex_pricing.validate_catalog(data)

    def test_offline_does_not_fetch_and_file_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp, patch('codex_pricing.urllib.request.urlopen') as fetch:
            file = Path(tmp) / 'pricing.json'
            codex_pricing.write_catalog(file, self.data)
            self.assertEqual(codex_pricing.load_catalog(file), self.data)
            with patch('codex_pricing.default_cache_path', return_value=Path(tmp) / 'absent.json'):
                self.assertTrue(codex_pricing.load_catalog(offline=True)['prices'])
            fetch.assert_not_called()

    def test_network_failure_falls_back_to_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp, patch('codex_pricing.default_cache_path', return_value=Path(tmp) / 'cache.json'), \
                patch('codex_pricing.urllib.request.urlopen', side_effect=OSError('offline')):
            with self.assertWarns(UserWarning):
                data = codex_pricing.load_catalog(refresh=True)
            self.assertTrue(data['prices'])

    def test_multi_request_delta_does_not_inherit_last_request_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'rollout-fixture.jsonl'
            log.write_text('\n'.join([
                turn_context('2026-10-01T01:00:00Z', 'gpt-6-sol'),
                token_event('2026-10-01T01:01:00Z', dict(input_tokens=600000, total_tokens=600000),
                            dict(input_tokens=300000, total_tokens=300000))]))
            report = codex_usage.build_report([Path(tmp)], self.stamp, self.stamp + timedelta(days=1),
                                              pricing=codex_pricing.Pricing(self.data))
        self.assertAlmostEqual(report.estimated_cost_usd, 1.2)
        self.assertIn('Missing request context size: short-context rates assumed', report.pricing['notes'])

    def test_parser_reads_exact_tiers_and_detects_format_drift(self):
        text = """### Standard pricing data
| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-sol | $2.00 | $0.20 | $2.50 | $10.00 | $4.00 | $0.40 | $5.00 | $15.00 |
### Fast pricing data
| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-sol | $4.00 | $0.40 | $5.00 | $20.00 | $8.00 | $0.80 | $10.00 | $30.00 |
Short context: ≤272K input tokens. Long context: >272K input tokens.
"""
        prices = codex_pricing.parse_prices(text)
        self.assertEqual(prices[('standard', 'gpt-6-sol')]['input'], 2)
        self.assertEqual(prices[('fast', 'gpt-6-sol')]['input'], 4)
        with self.assertRaises(ValueError):
            codex_pricing.parse_prices(text.replace('Short context input', 'Input'))


class UserMessageTests(unittest.TestCase):
    def message(self, timestamp, kind="legacy", item_id="message-1"):
        if kind == "legacy":
            payload = {"type": "user_message", "message": "Example request"}
        else:
            payload = {"type": "item_completed", "item": {"type": "UserMessage", "id": item_id}}
        return json.dumps({"timestamp": timestamp, "type": "event_msg", "payload": payload})

    def test_counts_both_log_formats_with_window_boundaries_and_deduplication(self):
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rollout-example.jsonl"
            for kind in ("legacy", "completed"):
                with self.subTest(kind=kind):
                    records = [
                        self.message("2026-05-31T23:59:59Z", kind, "before"),
                        self.message("2026-06-01T00:00:00Z", kind, "first"),
                        self.message("2026-06-01T12:00:00Z", kind, "second"),
                        self.message("2026-06-02T00:00:00Z", kind, "after"),
                        self.message("invalid", kind, "invalid"),
                        '{"user_message": broken',
                        json.dumps({"type": "response_item", "payload": {
                            "type": "message", "role": "user", "content": "user_message"}}),
                    ]
                    if kind == "completed":
                        records.extend([
                            self.message("2026-06-01T12:00:00Z", kind, "second"),
                            self.message("2026-06-01T12:00:00Z", "legacy"),
                            json.dumps({"timestamp": "2026-06-01T12:00:00Z", "type": "event_msg",
                                        "payload": {"type": "item_started", "item": {
                                            "type": "UserMessage", "id": "second"}}}),
                        ])
                    log.write_text("\n".join(records), encoding="utf-8")
                    self.assertEqual(codex_usage.count_user_messages(log, start, end), 2)

    def test_session_table_json_and_totals_use_message_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active, archived = root / "sessions", root / "archived_sessions"
            active.mkdir()
            archived.mkdir()
            for index, count in enumerate((3, 1)):
                name = f"rollout-{index}aaaaaaa-2222-3333-4444-555555555555.jsonl"
                content = "\n".join([
                    *(self.message(f"2026-06-01T0{hour}:00:00Z", item_id=str(hour))
                      for hour in range(count)),
                    token_event("2026-06-01T05:00:00Z", {"total_tokens": 100}),
                ])
                (active / name).write_text(content, encoding="utf-8")
                (archived / name).write_text(content, encoding="utf-8")
            report = codex_usage.build_report(
                [active, archived], datetime(2026, 6, 1, tzinfo=timezone.utc),
                datetime(2026, 6, 2, tzinfo=timezone.utc), group_by="session",
                sort_by="user-messages",
            )
        self.assertEqual([row.user_messages for row in report.rows], [3, 1])
        data = codex_usage.report_to_json(report)
        self.assertEqual(data["totals"]["user_messages"], 4)
        self.assertEqual(data["rows"][0]["user_messages"], 3)
        for mode, header in [("full", "User Messages"), ("compact", "Msgs")]:
            columns = codex_usage.table_columns(report, mode)
            self.assertEqual(columns[1].header, header)
            self.assertEqual(codex_usage.value_for_column(columns[1].key, report.rows[0]), "3")
            # Limiting displayed rows must retain the total across all sessions.
            total = codex_usage.total_row_for_columns(columns, report.rows[:1], report)
            self.assertEqual(total[1], "4")
        self.assertEqual(codex_usage.table_columns(replace(report, group_by="day"), "full")[1].header,
                         "Sessions")


class SortingTests(unittest.TestCase):
    def setUp(self):
        self.early = codex_usage.ReportRow(
            label="2026-06-01 01:00 aaaaaaaa · zebra", sessions=1,
            usage=codex_usage.Usage(input_tokens=10, cached_input_tokens=4,
                                   output_tokens=2, reasoning_output_tokens=1,
                                   total_tokens=12),
            session_name="zebra", first_event_at=datetime(2026, 6, 1, 1, tzinfo=timezone.utc),
            known_cost_usd=0.5, estimated_cost_usd=0.5, priced_events=1, user_messages=1,
        )
        self.late = codex_usage.ReportRow(
            label="2026-06-01 02:00 bbbbbbbb · Alpha", sessions=2,
            usage=codex_usage.Usage(input_tokens=20, cached_input_tokens=8,
                                   output_tokens=4, reasoning_output_tokens=2,
                                   total_tokens=24),
            session_name="Alpha", first_event_at=datetime(2026, 6, 1, 2, tzinfo=timezone.utc),
            known_cost_usd=1.0, estimated_cost_usd=1.0, priced_events=1, user_messages=2,
        )

    def test_time_and_name_use_metadata_instead_of_display_label(self):
        for field, ascending in [("time", [self.early, self.late]),
                                 ("name", [self.late, self.early])]:
            for descending in (False, True):
                with self.subTest(field=field, descending=descending):
                    rows = [self.late, self.early]
                    codex_usage.sort_report_rows(rows, "session", field, descending)
                    self.assertEqual(rows, list(reversed(ascending)) if descending else ascending)

    def test_numeric_fields_default_descending_and_allow_ascending(self):
        for field in set(codex_usage.SORT_FIELDS) - {"time", "name"}:
            with self.subTest(field=field):
                rows = [self.early, self.late]
                codex_usage.sort_report_rows(rows, "session", field)
                self.assertEqual(rows, [self.late, self.early])
                codex_usage.sort_report_rows(rows, "session", field, False)
                self.assertEqual(rows, [self.early, self.late])

    def test_default_order_is_preserved_and_direction_can_override_it(self):
        for group, descending, expected in [
            ("session", None, [self.late, self.early]),
            ("day", None, [self.early, self.late]),
            ("session", False, [self.early, self.late]),
            ("day", True, [self.late, self.early]),
        ]:
            rows = [self.late, self.early]
            codex_usage.sort_report_rows(rows, group, descending=descending)
            self.assertEqual(rows, expected)

    def test_cost_sorts_partial_subtotals_and_puts_unavailable_last(self):
        partial = replace(self.early, estimated_cost_usd=None, unpriced_events=1)
        unavailable = replace(self.late, label="Unavailable", known_cost_usd=0,
                              estimated_cost_usd=None, priced_events=0, unpriced_events=1)
        for descending, expected in [(True, [self.late, partial, unavailable]),
                                     (False, [partial, self.late, unavailable])]:
            rows = [unavailable, partial, self.late]
            codex_usage.sort_report_rows(rows, "session", "cost", descending)
            self.assertEqual(rows, expected)

    def test_unnamed_rows_use_label_and_empty_reports_sort(self):
        rows = [replace(self.early, session_name=None), replace(self.late, session_name=None)]
        codex_usage.sort_report_rows(rows, "model", "name")
        self.assertEqual(rows[0].label, self.early.label)
        for field in codex_usage.SORT_FIELDS:
            codex_usage.sort_report_rows([], "session", field)
        with self.assertRaises(ValueError):
            codex_usage.sort_report_rows([], "session", "invalid")

    def test_parser_accepts_directions_and_rejects_conflicts(self):
        parser = codex_usage.build_parser()
        self.assertIsNone(parser.parse_args([]).descending)
        for flag in ("--ascending", "--asc"):
            self.assertFalse(parser.parse_args([flag]).descending)
        for flag in ("--descending", "--desc"):
            self.assertTrue(parser.parse_args([flag]).descending)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--ascending", "--descending"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--sort-by", "invalid"])

    def test_cli_sorting_applies_to_json_and_precedes_table_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            sessions = home / "sessions"
            sessions.mkdir()
            ids = ["aaaaaaaa-2222-3333-4444-555555555555",
                   "bbbbbbbb-2222-3333-4444-555555555555"]
            for sid, hour, total in zip(ids, (1, 2), (10, 20)):
                (sessions / f"rollout-{sid}.jsonl").write_text(
                    token_event(f"2026-06-01T0{hour}:00:00Z", {"total_tokens": total}),
                    encoding="utf-8",
                )
            (home / "session_index.jsonl").write_text(
                "\n".join(json.dumps({"id": sid, "thread_name": name})
                          for sid, name in zip(ids, ("Zebra", "Alpha"))), encoding="utf-8",
            )
            args = ["--codex-home", tmp, "--since", "2026-06-01", "--until", "2026-06-01",
                    "--group-by", "session", "--offline"]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(codex_usage.main(args + ["--sort-by", "name", "--json"]), 0)
            report = json.loads(output.getvalue())
            self.assertIn("Alpha", report["rows"][0]["label"])
            self.assertEqual(report["totals"]["total_tokens"], 30)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                codex_usage.main(args + ["--sort-by", "time", "--desc", "--limit", "1"])
            self.assertIn("Alpha", output.getvalue())
            self.assertNotIn("Zebra", output.getvalue())


if __name__ == "__main__":
    unittest.main()
