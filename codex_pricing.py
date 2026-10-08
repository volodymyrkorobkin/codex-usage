"""Published OpenAI text prices with persistent, observation-based history."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCE_URL = 'https://developers.openai.com/api/docs/pricing'
TIERS = ('standard', 'batch', 'flex', 'fast', 'ultrafast')
RATE_KEYS = ('input', 'cached_input', 'cache_write', 'output',
             'long_input', 'long_cached_input', 'long_cache_write', 'long_output')


def utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Pricing dates must include a timezone')
    return result.astimezone(timezone.utc)


def parse_prices(markdown: str) -> dict:
    """Read only explicitly labelled text pricing tables; fail on format drift."""
    result = {}
    tier = None
    expected = ['Model', 'Short context input', 'Short context cached input',
                'Short context cache writes', 'Short context output',
                'Long context input', 'Long context cached input',
                'Long context cache writes', 'Long context output']
    table = False
    for line in markdown.splitlines():
        heading = re.fullmatch(r'### (Standard|Batch|Flex|Fast|Ultrafast) pricing data', line.strip())
        if heading:
            tier = heading[1].lower()
            table = False
        elif line.startswith('### '):
            tier = None
            table = False
        if not tier or not line.startswith('|'):
            continue
        cells = [part.strip() for part in line.strip().strip('|').split('|')]
        if cells[0] == 'Model':
            if cells != expected:
                raise ValueError('OpenAI pricing table columns changed')
            table = True
            continue
        if not table or cells[0].startswith('---'):
            continue
        if len(cells) != 9:
            raise ValueError('OpenAI pricing table row changed')
        model = re.sub(r'\s+\(.*\)$', '', cells[0])
        rates = {}
        for key, cell in zip(RATE_KEYS, cells[1:]):
            if cell == '-':
                rates[key] = None
            elif re.fullmatch(r'\$\d+(?:\.\d+)?', cell):
                rates[key] = float(cell[1:])
            else:
                raise ValueError(f'Invalid published rate for {model}: {cell}')
        result[(tier, model)] = rates
    if not result or not any(t == 'standard' for t, _ in result):
        raise ValueError('No Standard text pricing table found')
    if not re.search(r'Short context:.*272K.*Long context:.*272K', markdown):
        raise ValueError('OpenAI context pricing threshold changed')
    return result


def validate_catalog(data: dict) -> dict:
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('prices'), list):
        raise ValueError('Expected pricing schema_version 1 and a prices list')
    utc(data['checked_at'])
    groups = {}
    for item in data['prices']:
        if item['tier'] not in TIERS or not item['model'] or not item.get('source'):
            raise ValueError('Invalid pricing model, tier or source')
        start = utc(item['start'])
        end = utc(item['end']) if item.get('end') else None
        if end is not None and end <= start:
            raise ValueError('Pricing end must be after start')
        if item.get('date_basis') not in ('observed', 'effective'):
            raise ValueError('date_basis must be observed or effective')
        if item.get('long_context_threshold') != 272000:
            raise ValueError('Unsupported context threshold')
        for key in RATE_KEYS:
            rate = item['rates'][key]
            if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate < 0):
                raise ValueError('Rates must be finite nonnegative numbers or null')
        if item['rates']['input'] is None or item['rates']['output'] is None:
            raise ValueError('Input and output rates are required')
        groups.setdefault((item['tier'], item['model']), []).append((start, end))
    for periods in groups.values():
        periods.sort()
        for (_, end), (start, _) in zip(periods, periods[1:]):
            if end is None or end > start:
                raise ValueError('Overlapping pricing periods')
    return data


def update_catalog(data: dict, prices: dict, now: datetime) -> dict:
    # A changed price begins when observed; never invent its actual effective date.
    data = json.loads(json.dumps(data))
    stamp = now.isoformat()
    for (tier, model), rates in prices.items():
        active = next((p for p in reversed(data['prices'])
                       if p['tier'] == tier and p['model'] == model and p['end'] is None), None)
        if active and active['rates'] == rates:
            continue
        if active:
            active['end'] = stamp
        data['prices'].append(dict(model=model, tier=tier, start=stamp, end=None,
                                   date_basis='observed', source=SOURCE_URL,
                                   long_context_threshold=272000, rates=rates))
    data['checked_at'] = stamp
    return validate_catalog(data)


def write_catalog(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement keeps interrupted updates from corrupting the cache.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        json.dump(data, handle, indent=2)
        handle.write('\n')
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def default_cache_path() -> Path:
    base = Path(os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    return base / 'codex-usage' / 'pricing.json'


def load_catalog(path: Path | None = None, offline: bool = False, refresh: bool = False) -> dict:
    if path is not None:
        return validate_catalog(json.loads(path.expanduser().read_text(encoding='utf-8')))
    cache = default_cache_path()
    data = bundled_catalog()
    if cache.exists():
        try:
            data = validate_catalog(json.loads(cache.read_text(encoding='utf-8')))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            warnings.warn(f'Cannot read pricing cache; using bundled snapshot: {exc}')
    now = datetime.now(timezone.utc)
    if not offline and (refresh or now - utc(data['checked_at']) >= timedelta(days=1) or not cache.exists()):
        try:
            request = urllib.request.Request(SOURCE_URL + '.md', headers={'User-Agent': 'codex-usage/0.2'})
            with urllib.request.urlopen(request, timeout=10) as response:
                markdown = response.read(2_000_000).decode('utf-8')
            data = update_catalog(data, parse_prices(markdown), now)
            write_catalog(cache, data)
        except (OSError, ValueError) as exc:
            warnings.warn(f'Cannot refresh OpenAI prices; using snapshot from {data["checked_at"]}: {exc}')
    return data


class Pricing:
    def __init__(self, data: dict, tier: str = 'standard', strict_history: bool = False):
        self.data = validate_catalog(data)
        self.tier = tier
        self.strict_history = strict_history
        self.used: dict[str, dict] = {}
        self.notes: set[str] = set()
        self.fallback_models: dict[str, dict] = {}

    def _price_at(self, model: str, timestamp: datetime):
        candidates = [p for p in self.data['prices'] if p['model'] == model and p['tier'] == self.tier]
        matched = [p for p in candidates if utc(p['start']) <= timestamp and
                   (p['end'] is None or timestamp < utc(p['end']))]
        if matched:
            return matched[0], False
        elif candidates and not self.strict_history and timestamp < min(utc(p['start']) for p in candidates):
            return max(candidates, key=lambda p: utc(p['start'])), True
        return None, False

    @staticmethod
    def _amount(usage, price, request_input):
        prefix = 'long_' if request_input is not None and request_input > price['long_context_threshold'] else ''
        rates = price['rates']
        input_rate, cached_rate, output_rate = (rates[prefix + key] for key in ('input', 'cached_input', 'output'))
        if input_rate is None or output_rate is None or (cached_rate is None and usage.cached_input_tokens):
            return None
        return (usage.uncached_input_tokens * input_rate + usage.cached_input_tokens * (cached_rate or 0) + usage.output_tokens * output_rate) / 1_000_000

    def cost(self, usage, model: str, timestamp: datetime, request_input: int | None) -> float | None:
        price, historical_fallback = self._price_at(model, timestamp)
        # Only unknown model names use a proxy. Known models retain missing
        # tier/context/history errors rather than silently changing their rate.
        unknown = not any(p['model'] == model for p in self.data['prices'])
        if price is None and unknown:
            choices = []
            available_models = {p['model'] for p in self.data['prices'] if p['tier'] == self.tier}
            # Compare Codex model families, not unrelated legacy/API-only
            # models such as GPT-5 Nano. Custom catalogs without these families
            # use their own complete model list.
            codex_models = {name for name in available_models
                            if name == 'gpt-5.5' or re.fullmatch(
                                r'gpt-\d+(?:\.\d+)?-(?:luna|sol|terra|astra|codex)(?:-.*)?', name)}
            for candidate_model in sorted(codex_models or available_models):
                candidate, historical = self._price_at(candidate_model, timestamp)
                if candidate is not None:
                    amount = self._amount(usage, candidate, request_input)
                    if amount is not None:
                        choices.append((amount, candidate['rates']['input'], candidate_model, candidate, historical))
            if choices:
                _, _, _, price, historical_fallback = min(choices, key=lambda c: c[:3])
        if price is None:
            self.notes.add(f'No price for {model} ({self.tier}) at event time')
            return None
        amount = self._amount(usage, price, request_input)
        if amount is None:
            self.notes.add(f'No applicable context/cache rate for {model}')
            return None
        if unknown:
            self.notes.add(f'Estimated {model} using cheapest available Codex model: {price["model"]} ({self.tier})')
            key = json.dumps([model, price['model']])
            record = self.fallback_models.setdefault(key, dict(model=model, priced_as=price['model'], events=0, estimated_cost_usd=0.0))
            record['events'] += 1
            record['estimated_cost_usd'] += amount
        if historical_fallback:
            self.notes.add('Current published rates used before recorded history')
        self.used[json.dumps(price, sort_keys=True)] = price
        if price['date_basis'] == 'observed':
            self.notes.add('Price dates are observation dates, not verified effective dates')
        if request_input is None:
            self.notes.add('Missing request context size: short-context rates assumed')
        return amount

    def as_dict(self) -> dict:
        return dict(source=SOURCE_URL, checked_at=self.data['checked_at'], tier=self.tier,
                    strict_history=self.strict_history, rates=list(self.used.values()),
                    fallback_models=list(self.fallback_models.values()),
                    notes=sorted(self.notes),
                    note='API-equivalent text cost; ChatGPT-auth Codex usage is not API billing. Cache writes, tools and regional surcharges are not included.')


# Official pricing snapshot fetched 2026-10-08T08:17:51.771409+00:00; refreshed automatically at runtime.
BUNDLED_CHECKED_AT = '2026-10-08T08:17:51.771409+00:00'
BUNDLED_RATES = {
    ('standard', 'gpt-6-astra'): {'input': 10.0, 'cached_input': 1.0, 'cache_write': 12.5, 'output': 50.0, 'long_input': 20.0, 'long_cached_input': 2.0, 'long_cache_write': 25.0, 'long_output': 75.0},
    ('standard', 'gpt-6.1-sol'): {'input': 2.0, 'cached_input': 0.1, 'cache_write': 2.5, 'output': 10.0, 'long_input': 4.0, 'long_cached_input': 0.2, 'long_cache_write': 5.0, 'long_output': 15.0},
    ('standard', 'gpt-6-luna'): {'input': 0.1, 'cached_input': 0.01, 'cache_write': 0.125, 'output': 0.5, 'long_input': 0.2, 'long_cached_input': 0.02, 'long_cache_write': 0.25, 'long_output': 0.75},
    ('standard', 'gpt-6-sol'): {'input': 2.0, 'cached_input': 0.2, 'cache_write': 2.5, 'output': 10.0, 'long_input': 4.0, 'long_cached_input': 0.4, 'long_cache_write': 5.0, 'long_output': 15.0},
    ('standard', 'gpt-5.6-sol'): {'input': 4.0, 'cached_input': 0.4, 'cache_write': 5.0, 'output': 20.0, 'long_input': 8.0, 'long_cached_input': 0.8, 'long_cache_write': 10.0, 'long_output': 30.0},
    ('standard', 'gpt-5.6-terra'): {'input': 2.0, 'cached_input': 0.2, 'cache_write': 2.5, 'output': 12.0, 'long_input': 4.0, 'long_cached_input': 0.4, 'long_cache_write': 5.0, 'long_output': 18.0},
    ('standard', 'gpt-5.6-luna'): {'input': 0.2, 'cached_input': 0.02, 'cache_write': 0.25, 'output': 1.2, 'long_input': 0.4, 'long_cached_input': 0.04, 'long_cache_write': 0.5, 'long_output': 1.8},
    ('standard', 'gpt-5.5'): {'input': 5.0, 'cached_input': 0.5, 'cache_write': None, 'output': 30.0, 'long_input': 10.0, 'long_cached_input': 1.0, 'long_cache_write': None, 'long_output': 45.0},
    ('standard', 'gpt-5.5-pro'): {'input': 30.0, 'cached_input': None, 'cache_write': None, 'output': 180.0, 'long_input': 60.0, 'long_cached_input': None, 'long_cache_write': None, 'long_output': 270.0},
    ('standard', 'gpt-5.4'): {'input': 2.5, 'cached_input': 0.25, 'cache_write': None, 'output': 15.0, 'long_input': 5.0, 'long_cached_input': 0.5, 'long_cache_write': None, 'long_output': 22.5},
    ('standard', 'gpt-5.4-mini'): {'input': 0.75, 'cached_input': 0.075, 'cache_write': None, 'output': 4.5, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-5.4-nano'): {'input': 0.2, 'cached_input': 0.02, 'cache_write': None, 'output': 1.25, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-5.4-pro'): {'input': 30.0, 'cached_input': None, 'cache_write': None, 'output': 180.0, 'long_input': 60.0, 'long_cached_input': None, 'long_cache_write': None, 'long_output': 270.0},
    ('standard', 'gpt-5.2'): {'input': 1.75, 'cached_input': 0.175, 'cache_write': None, 'output': 14.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-5.2-pro'): {'input': 21.0, 'cached_input': None, 'cache_write': None, 'output': 168.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-5.1'): {'input': 1.25, 'cached_input': 0.125, 'cache_write': None, 'output': 10.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-5'): {'input': 1.25, 'cached_input': 0.125, 'cache_write': None, 'output': 10.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-5-mini'): {'input': 0.25, 'cached_input': 0.025, 'cache_write': None, 'output': 2.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-5-nano'): {'input': 0.05, 'cached_input': 0.005, 'cache_write': None, 'output': 0.4, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-5-pro'): {'input': 15.0, 'cached_input': None, 'cache_write': None, 'output': 120.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-4.1'): {'input': 2.0, 'cached_input': 0.5, 'cache_write': None, 'output': 8.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-4.1-mini'): {'input': 0.4, 'cached_input': 0.1, 'cache_write': None, 'output': 1.6, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-4.1-nano'): {'input': 0.1, 'cached_input': 0.025, 'cache_write': None, 'output': 0.4, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-4o'): {'input': 2.5, 'cached_input': 1.25, 'cache_write': None, 'output': 10.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-4o-2024-05-13'): {'input': 5.0, 'cached_input': None, 'cache_write': None, 'output': 15.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-4o-mini'): {'input': 0.15, 'cached_input': 0.075, 'cache_write': None, 'output': 0.6, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'o1'): {'input': 15.0, 'cached_input': 7.5, 'cache_write': None, 'output': 60.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'o1-pro'): {'input': 150.0, 'cached_input': None, 'cache_write': None, 'output': 600.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'o3-pro'): {'input': 20.0, 'cached_input': None, 'cache_write': None, 'output': 80.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'o3'): {'input': 2.0, 'cached_input': 0.5, 'cache_write': None, 'output': 8.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'o4-mini'): {'input': 1.1, 'cached_input': 0.275, 'cache_write': None, 'output': 4.4, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'o3-mini'): {'input': 1.1, 'cached_input': 0.55, 'cache_write': None, 'output': 4.4, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-4-turbo-2024-04-09'): {'input': 10.0, 'cached_input': None, 'cache_write': None, 'output': 30.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-4-0613'): {'input': 30.0, 'cached_input': None, 'cache_write': None, 'output': 60.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-3.5-turbo'): {'input': 0.5, 'cached_input': None, 'cache_write': None, 'output': 1.5, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-3.5-turbo-0125'): {'input': 0.5, 'cached_input': None, 'cache_write': None, 'output': 1.5, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-3.5-turbo-1106'): {'input': 1.0, 'cached_input': None, 'cache_write': None, 'output': 2.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'gpt-3.5-turbo-instruct'): {'input': 1.5, 'cached_input': None, 'cache_write': None, 'output': 2.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'davinci-002'): {'input': 2.0, 'cached_input': None, 'cache_write': None, 'output': 2.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('standard', 'babbage-002'): {'input': 0.4, 'cached_input': None, 'cache_write': None, 'output': 0.4, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-6-astra'): {'input': 5.0, 'cached_input': 0.5, 'cache_write': 6.25, 'output': 25.0, 'long_input': 10.0, 'long_cached_input': 1.0, 'long_cache_write': 12.5, 'long_output': 37.5},
    ('batch', 'gpt-6.1-sol'): {'input': 1.0, 'cached_input': 0.05, 'cache_write': 1.25, 'output': 5.0, 'long_input': 2.0, 'long_cached_input': 0.1, 'long_cache_write': 2.5, 'long_output': 7.5},
    ('batch', 'gpt-6-luna'): {'input': 0.05, 'cached_input': 0.005, 'cache_write': 0.0625, 'output': 0.25, 'long_input': 0.1, 'long_cached_input': 0.01, 'long_cache_write': 0.125, 'long_output': 0.375},
    ('batch', 'gpt-6-sol'): {'input': 1.0, 'cached_input': 0.1, 'cache_write': 1.25, 'output': 5.0, 'long_input': 2.0, 'long_cached_input': 0.2, 'long_cache_write': 2.5, 'long_output': 7.5},
    ('batch', 'gpt-5.6-sol'): {'input': 2.0, 'cached_input': 0.2, 'cache_write': 2.5, 'output': 10.0, 'long_input': 4.0, 'long_cached_input': 0.4, 'long_cache_write': 5.0, 'long_output': 15.0},
    ('batch', 'gpt-5.6-terra'): {'input': 1.0, 'cached_input': 0.1, 'cache_write': 1.25, 'output': 6.0, 'long_input': 2.0, 'long_cached_input': 0.2, 'long_cache_write': 2.5, 'long_output': 9.0},
    ('batch', 'gpt-5.6-luna'): {'input': 0.1, 'cached_input': 0.01, 'cache_write': 0.125, 'output': 0.6, 'long_input': 0.2, 'long_cached_input': 0.02, 'long_cache_write': 0.25, 'long_output': 0.9},
    ('batch', 'gpt-5.5'): {'input': 2.5, 'cached_input': 0.25, 'cache_write': None, 'output': 15.0, 'long_input': 5.0, 'long_cached_input': 0.5, 'long_cache_write': None, 'long_output': 22.5},
    ('batch', 'gpt-5.5-pro'): {'input': 15.0, 'cached_input': None, 'cache_write': None, 'output': 90.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5.4'): {'input': 1.25, 'cached_input': 0.13, 'cache_write': None, 'output': 7.5, 'long_input': 2.5, 'long_cached_input': 0.25, 'long_cache_write': None, 'long_output': 11.25},
    ('batch', 'gpt-5.4-mini'): {'input': 0.375, 'cached_input': 0.0375, 'cache_write': None, 'output': 2.25, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5.4-nano'): {'input': 0.1, 'cached_input': 0.01, 'cache_write': None, 'output': 0.625, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5.4-pro'): {'input': 15.0, 'cached_input': None, 'cache_write': None, 'output': 90.0, 'long_input': 30.0, 'long_cached_input': None, 'long_cache_write': None, 'long_output': 135.0},
    ('batch', 'gpt-5.2'): {'input': 0.875, 'cached_input': 0.0875, 'cache_write': None, 'output': 7.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5.2-pro'): {'input': 10.5, 'cached_input': None, 'cache_write': None, 'output': 84.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5.1'): {'input': 0.625, 'cached_input': 0.0625, 'cache_write': None, 'output': 5.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5'): {'input': 0.625, 'cached_input': 0.0625, 'cache_write': None, 'output': 5.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5-mini'): {'input': 0.125, 'cached_input': 0.0125, 'cache_write': None, 'output': 1.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5-nano'): {'input': 0.025, 'cached_input': 0.0025, 'cache_write': None, 'output': 0.2, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-5-pro'): {'input': 7.5, 'cached_input': None, 'cache_write': None, 'output': 60.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-4.1'): {'input': 1.0, 'cached_input': None, 'cache_write': None, 'output': 4.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-4.1-mini'): {'input': 0.2, 'cached_input': None, 'cache_write': None, 'output': 0.8, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-4.1-nano'): {'input': 0.05, 'cached_input': None, 'cache_write': None, 'output': 0.2, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-4o'): {'input': 1.25, 'cached_input': None, 'cache_write': None, 'output': 5.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-4o-2024-05-13'): {'input': 2.5, 'cached_input': None, 'cache_write': None, 'output': 7.5, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-4o-mini'): {'input': 0.075, 'cached_input': None, 'cache_write': None, 'output': 0.3, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'o1'): {'input': 7.5, 'cached_input': None, 'cache_write': None, 'output': 30.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'o1-pro'): {'input': 75.0, 'cached_input': None, 'cache_write': None, 'output': 300.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'o3-pro'): {'input': 10.0, 'cached_input': None, 'cache_write': None, 'output': 40.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'o3'): {'input': 1.0, 'cached_input': None, 'cache_write': None, 'output': 4.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'o4-mini'): {'input': 0.55, 'cached_input': None, 'cache_write': None, 'output': 2.2, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'o3-mini'): {'input': 0.55, 'cached_input': None, 'cache_write': None, 'output': 2.2, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-4-turbo-2024-04-09'): {'input': 5.0, 'cached_input': None, 'cache_write': None, 'output': 15.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-4-0613'): {'input': 15.0, 'cached_input': None, 'cache_write': None, 'output': 30.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-3.5-turbo-0125'): {'input': 0.25, 'cached_input': None, 'cache_write': None, 'output': 0.75, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'gpt-3.5-turbo-1106'): {'input': 1.0, 'cached_input': None, 'cache_write': None, 'output': 2.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'davinci-002'): {'input': 1.0, 'cached_input': None, 'cache_write': None, 'output': 1.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('batch', 'babbage-002'): {'input': 0.2, 'cached_input': None, 'cache_write': None, 'output': 0.2, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'gpt-6-astra'): {'input': 5.0, 'cached_input': 0.5, 'cache_write': 6.25, 'output': 25.0, 'long_input': 10.0, 'long_cached_input': 1.0, 'long_cache_write': 12.5, 'long_output': 37.5},
    ('flex', 'gpt-6.1-sol'): {'input': 1.0, 'cached_input': 0.05, 'cache_write': 1.25, 'output': 5.0, 'long_input': 2.0, 'long_cached_input': 0.1, 'long_cache_write': 2.5, 'long_output': 7.5},
    ('flex', 'gpt-6-luna'): {'input': 0.05, 'cached_input': 0.005, 'cache_write': 0.0625, 'output': 0.25, 'long_input': 0.1, 'long_cached_input': 0.01, 'long_cache_write': 0.125, 'long_output': 0.375},
    ('flex', 'gpt-6-sol'): {'input': 1.0, 'cached_input': 0.1, 'cache_write': 1.25, 'output': 5.0, 'long_input': 2.0, 'long_cached_input': 0.2, 'long_cache_write': 2.5, 'long_output': 7.5},
    ('flex', 'gpt-5.6-sol'): {'input': 2.0, 'cached_input': 0.2, 'cache_write': 2.5, 'output': 10.0, 'long_input': 4.0, 'long_cached_input': 0.4, 'long_cache_write': 5.0, 'long_output': 15.0},
    ('flex', 'gpt-5.6-terra'): {'input': 1.0, 'cached_input': 0.1, 'cache_write': 1.25, 'output': 6.0, 'long_input': 2.0, 'long_cached_input': 0.2, 'long_cache_write': 2.5, 'long_output': 9.0},
    ('flex', 'gpt-5.6-luna'): {'input': 0.1, 'cached_input': 0.01, 'cache_write': 0.125, 'output': 0.6, 'long_input': 0.2, 'long_cached_input': 0.02, 'long_cache_write': 0.25, 'long_output': 0.9},
    ('flex', 'gpt-5.5'): {'input': 2.5, 'cached_input': 0.25, 'cache_write': None, 'output': 15.0, 'long_input': 5.0, 'long_cached_input': 0.5, 'long_cache_write': None, 'long_output': 22.5},
    ('flex', 'gpt-5.5-pro'): {'input': 15.0, 'cached_input': None, 'cache_write': None, 'output': 90.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'gpt-5.4'): {'input': 1.25, 'cached_input': 0.13, 'cache_write': None, 'output': 7.5, 'long_input': 2.5, 'long_cached_input': 0.25, 'long_cache_write': None, 'long_output': 11.25},
    ('flex', 'gpt-5.4-mini'): {'input': 0.375, 'cached_input': 0.0375, 'cache_write': None, 'output': 2.25, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'gpt-5.4-nano'): {'input': 0.1, 'cached_input': 0.01, 'cache_write': None, 'output': 0.625, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'gpt-5.4-pro'): {'input': 15.0, 'cached_input': None, 'cache_write': None, 'output': 90.0, 'long_input': 30.0, 'long_cached_input': None, 'long_cache_write': None, 'long_output': 135.0},
    ('flex', 'gpt-5.2'): {'input': 0.875, 'cached_input': 0.0875, 'cache_write': None, 'output': 7.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'gpt-5.1'): {'input': 0.625, 'cached_input': 0.0625, 'cache_write': None, 'output': 5.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'gpt-5'): {'input': 0.625, 'cached_input': 0.0625, 'cache_write': None, 'output': 5.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'gpt-5-mini'): {'input': 0.125, 'cached_input': 0.0125, 'cache_write': None, 'output': 1.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'gpt-5-nano'): {'input': 0.025, 'cached_input': 0.0025, 'cache_write': None, 'output': 0.2, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'o3'): {'input': 1.0, 'cached_input': 0.25, 'cache_write': None, 'output': 4.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('flex', 'o4-mini'): {'input': 0.55, 'cached_input': 0.138, 'cache_write': None, 'output': 2.2, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-6-astra'): {'input': 20.0, 'cached_input': 2.0, 'cache_write': 25.0, 'output': 100.0, 'long_input': 40.0, 'long_cached_input': 4.0, 'long_cache_write': 50.0, 'long_output': 150.0},
    ('fast', 'gpt-6.1-sol'): {'input': 4.0, 'cached_input': 0.2, 'cache_write': 5.0, 'output': 20.0, 'long_input': 8.0, 'long_cached_input': 0.4, 'long_cache_write': 10.0, 'long_output': 30.0},
    ('fast', 'gpt-6-luna'): {'input': 0.2, 'cached_input': 0.02, 'cache_write': 0.25, 'output': 1.0, 'long_input': 0.4, 'long_cached_input': 0.04, 'long_cache_write': 0.5, 'long_output': 1.5},
    ('fast', 'gpt-6-sol'): {'input': 4.0, 'cached_input': 0.4, 'cache_write': 5.0, 'output': 20.0, 'long_input': 8.0, 'long_cached_input': 0.8, 'long_cache_write': 10.0, 'long_output': 30.0},
    ('fast', 'gpt-5.6-sol'): {'input': 8.0, 'cached_input': 0.8, 'cache_write': 10.0, 'output': 40.0, 'long_input': 16.0, 'long_cached_input': 1.6, 'long_cache_write': 20.0, 'long_output': 60.0},
    ('fast', 'gpt-5.6-terra'): {'input': 4.0, 'cached_input': 0.4, 'cache_write': 5.0, 'output': 24.0, 'long_input': 8.0, 'long_cached_input': 0.8, 'long_cache_write': 10.0, 'long_output': 36.0},
    ('fast', 'gpt-5.6-luna'): {'input': 0.4, 'cached_input': 0.04, 'cache_write': 0.5, 'output': 2.4, 'long_input': 0.8, 'long_cached_input': 0.08, 'long_cache_write': 1.0, 'long_output': 3.6},
    ('fast', 'gpt-5.5'): {'input': 12.5, 'cached_input': 1.25, 'cache_write': None, 'output': 75.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-5.4'): {'input': 5.0, 'cached_input': 0.5, 'cache_write': None, 'output': 30.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-5.4-mini'): {'input': 1.5, 'cached_input': 0.15, 'cache_write': None, 'output': 9.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-5.2'): {'input': 3.5, 'cached_input': 0.35, 'cache_write': None, 'output': 28.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-5.1'): {'input': 2.5, 'cached_input': 0.25, 'cache_write': None, 'output': 20.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-5'): {'input': 2.5, 'cached_input': 0.25, 'cache_write': None, 'output': 20.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-5-mini'): {'input': 0.45, 'cached_input': 0.045, 'cache_write': None, 'output': 3.6, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-4.1'): {'input': 3.5, 'cached_input': 0.875, 'cache_write': None, 'output': 14.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-4.1-mini'): {'input': 0.7, 'cached_input': 0.175, 'cache_write': None, 'output': 2.8, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-4.1-nano'): {'input': 0.2, 'cached_input': 0.05, 'cache_write': None, 'output': 0.8, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-4o'): {'input': 4.25, 'cached_input': 2.125, 'cache_write': None, 'output': 17.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-4o-2024-05-13'): {'input': 8.75, 'cached_input': None, 'cache_write': None, 'output': 26.25, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'gpt-4o-mini'): {'input': 0.25, 'cached_input': 0.125, 'cache_write': None, 'output': 1.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'o3'): {'input': 3.5, 'cached_input': 0.875, 'cache_write': None, 'output': 14.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('fast', 'o4-mini'): {'input': 2.0, 'cached_input': 0.5, 'cache_write': None, 'output': 8.0, 'long_input': None, 'long_cached_input': None, 'long_cache_write': None, 'long_output': None},
    ('ultrafast', 'gpt-6-astra'): {'input': 60.0, 'cached_input': 6.0, 'cache_write': 75.0, 'output': 300.0, 'long_input': 120.0, 'long_cached_input': 12.0, 'long_cache_write': 150.0, 'long_output': 450.0},
}


def bundled_catalog() -> dict:
    return dict(schema_version=1, checked_at=BUNDLED_CHECKED_AT, prices=[
        dict(model=model, tier=tier, start=BUNDLED_CHECKED_AT, end=None,
             date_basis="observed", source=SOURCE_URL, long_context_threshold=272000, rates=rates.copy())
        for (tier, model), rates in BUNDLED_RATES.items()
    ])
