from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import codex_usage
import codex_pricing
from unittest.mock import patch
from datetime import timedelta


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
        self.assertIsNone(pricing.cost(usage, 'unknown', self.stamp, 10))
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

    def test_missing_model_marks_row_and_total_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'rollout-fixture.jsonl'
            log.write_text(token_event('2026-10-01T01:00:00Z', dict(input_tokens=10, total_tokens=10)))
            report = codex_usage.build_report([Path(tmp)], self.stamp, self.stamp + timedelta(days=1),
                                              pricing=codex_pricing.Pricing(self.data))
        self.assertEqual(report.unpriced_events, 1)
        self.assertIsNone(report.estimated_cost_usd)
        self.assertIsNone(codex_usage.report_to_json(report)['rows'][0]['estimated_cost_usd'])

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
        self.assertEqual(daily.unpriced_events, 1)
        self.assertEqual(daily.priced_events, 1)
        self.assertIsNone(daily.estimated_cost_usd)
        self.assertAlmostEqual(daily.known_cost_usd, 0.2)
        self.assertEqual(codex_usage.display_cost(daily.rows[0]), '$0.20*')
        for mode in ('full', 'compact', 'auto'):
            rendered = codex_usage.render_report(daily, color=False, table_mode=mode)
            self.assertIn('$0.20*', rendered)
            self.assertIn('Partial cost', rendered)
        rows = {row.label: row for row in models.rows}
        self.assertEqual(codex_usage.display_cost(rows['codex-auto-review']), 'N/A')
        self.assertEqual(codex_usage.display_cost(rows['gpt-6-sol']), '$0.20')
        self.assertIn('$0.20*', codex_usage.render_report(models, color=False, limit=1))
        data = codex_usage.report_to_json(daily)
        self.assertEqual(data['rows'][0]['known_cost_usd'], 0.2)
        self.assertEqual(data['totals']['known_cost_usd'], 0.2)
        self.assertEqual(data['totals']['priced_events'], 1)
        self.assertIsNone(data['totals']['estimated_cost_usd'])

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


if __name__ == "__main__":
    unittest.main()
