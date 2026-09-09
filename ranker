"""
ranker.py - Momentum ranking engine (Phase 1 of concentration strategy).

Purpose
-------
Every 4 hours, score a universe of large-cap USDT pairs on multi-factor
momentum. Output ranked list to /data/ranked_pairs.json for the dashboard.

Phase 1 is READ-ONLY. It never places trades. The existing SignalEngine +
Trader continue to run untouched. This is pure observation - the user sees
what the new strategy WOULD pick, alongside what the current bot IS doing.

Scoring model (100-point scale)
--------------------------------
1. Momentum (0-40)   - 30d price change, scaled and clipped
2. Trend (0-25)      - price vs 50d MA, boosted if in strong uptrend
3. Volume (0-15)     - 7d volume vs 30d volume (rising interest)
4. Relative (0-15)   - price change vs BTC over 30d (alpha detection)
5. Consistency (0-5) - Sharpe-like: mean daily return / stdev

BTC gets a permanent +15 point bias so we don't rotate into weaker alts
just because they're rallying harder short-term. BTC is the anchor.

Design notes
------------
- Uses only Binance klines - no external APIs. Cheap and reliable.
- Runs on a background thread from app.py, 4h interval.
- Persists to /data/ranked_pairs.json - survives restarts, dashboard reads.
- Universe hardcoded to ~40 large-cap pairs. Can be tuned later.
- All Binance API calls wrapped in try/except - one bad ticker doesn't kill
  the whole run.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime

log = logging.getLogger(__name__)

_RANKING_PATH = '/data/ranked_pairs.json'
_RANKING_INTERVAL_SECONDS = 4 * 60 * 60  # 4 hours

# BTC gets a score bias because it's the anchor asset. We don't want to
# rotate into a random alt just because it pumped 20% overnight while BTC
# was flat - that's usually a chase trade. BTC has to be materially weaker
# than an alt for the alt to win.
BTC_SCORE_BONUS = 15.0

# Universe - large-cap USDT pairs with sufficient liquidity and history.
# Curated list; we don't blindly scan Binance because that would pull in
# tokenized stocks, low-cap garbage, and stablecoin pairs.
UNIVERSE = [
    # Tier 1 - majors (highest liquidity, most reliable data)
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
    # Tier 2 - established L1s / L2s
    'AVAXUSDT', 'ADAUSDT', 'DOTUSDT', 'MATICUSDT', 'ATOMUSDT',
    'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'ARBUSDT', 'OPUSDT',
    # Tier 3 - blue-chip DeFi / infra
    'LINKUSDT', 'UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'INJUSDT',
    # Tier 4 - large-cap memes / high-momentum
    'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT',
    # Tier 5 - AI / trending narratives
    'FETUSDT', 'RENDERUSDT', 'TAOUSDT', 'ONDOUSDT',
    # Tier 6 - other majors worth watching
    'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'ETCUSDT', 'FILUSDT',
    'TIAUSDT', 'SEIUSDT', 'JUPUSDT', 'PYTHUSDT',
]


def _pct(a, b):
    """Percent change from a to b. Returns 0 if a is zero/negative."""
    if not a or a <= 0:
        return 0.0
    return ((b - a) / a) * 100.0


def _score_momentum(price_30d_ago, price_now):
    """0-40 points. Uses 30d price change, sqrt-scaled to reward strong
    trends without letting outliers dominate the ranking entirely."""
    change = _pct(price_30d_ago, price_now)
    if change <= 0:
        return 0.0
    # Sqrt scale: 25% change -> 20 pts, 100% change -> 40 pts (capped)
    scaled = (change ** 0.5) * 4.0
    return min(40.0, scaled)


def _score_trend(closes, ma_period=50):
    """0-25 points. Distance above MA + slope of MA."""
    if len(closes) < ma_period + 5:
        return 0.0
    ma_now = sum(closes[-ma_period:]) / ma_period
    ma_prev = sum(closes[-ma_period - 5:-5]) / ma_period
    price = closes[-1]
    if ma_now <= 0:
        return 0.0
    # Points for price above MA (0-15)
    distance_pct = _pct(ma_now, price)
    distance_score = min(15.0, max(0.0, distance_pct * 0.75))
    # Points for MA rising (0-10)
    slope_pct = _pct(ma_prev, ma_now)
    slope_score = min(10.0, max(0.0, slope_pct * 2.0))
    return distance_score + slope_score


def _score_volume(volumes):
    """0-15 points. Recent 7d avg volume vs 30d avg volume. Rising volume
    on a coin often precedes price moves - it's a leading indicator."""
    if len(volumes) < 30:
        return 0.0
    vol_7d = sum(volumes[-7:]) / 7
    vol_30d = sum(volumes[-30:]) / 30
    if vol_30d <= 0:
        return 0.0
    ratio = vol_7d / vol_30d
    if ratio < 1.0:
        return 0.0  # Volume falling - skip
    # 1.0x -> 0, 1.5x -> 7.5, 2x+ -> 15
    return min(15.0, (ratio - 1.0) * 15.0)


def _score_relative_to_btc(coin_change_30d, btc_change_30d):
    """0-15 points. How much did this coin outperform BTC over 30d?
    This is the alpha signal - what's genuinely leading the market."""
    if coin_change_30d <= btc_change_30d:
        return 0.0
    outperformance = coin_change_30d - btc_change_30d
    # 10pp outperformance -> 5, 30pp -> 15
    return min(15.0, outperformance * 0.5)


def _score_consistency(closes):
    """0-5 points. Sharpe-like: mean daily return / stdev. Rewards smooth
    uptrends over choppy ones with the same net gain."""
    if len(closes) < 15:
        return 0.0
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
    if len(returns) < 10:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    stdev = variance ** 0.5
    if stdev <= 0:
        return 0.0
    sharpe_like = mean / stdev
    # Clip: sharpe of 0.1 -> 2.5, 0.2+ -> 5
    return max(0.0, min(5.0, sharpe_like * 25))


def _fetch_klines(client, symbol, interval='1d', limit=60):
    """Wrapper around get_klines with a try/except. Returns list of
    [open_time, open, high, low, close, volume, ...] rows, or None."""
    try:
        return client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        log.debug(f"kline fetch failed for {symbol}: {e}")
        return None


def score_pair(client, symbol, btc_change_30d):
    """Compute a full score for one symbol. Returns dict or None on failure.

    btc_change_30d is passed in so we don't refetch it 40 times.
    """
    klines = _fetch_klines(client, symbol, interval='1d', limit=60)
    if not klines or len(klines) < 30:
        return None
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    price_now = closes[-1]
    price_30d_ago = closes[-30]

    momentum = _score_momentum(price_30d_ago, price_now)
    trend = _score_trend(closes)
    volume = _score_volume(volumes)
    coin_change_30d = _pct(price_30d_ago, price_now)
    relative = _score_relative_to_btc(coin_change_30d, btc_change_30d)
    consistency = _score_consistency(closes)

    total = momentum + trend + volume + relative + consistency
    if symbol == 'BTCUSDT':
        total += BTC_SCORE_BONUS

    return {
        'symbol': symbol,
        'display': symbol.replace('USDT', '/USDT'),
        'price': price_now,
        'change_30d_pct': round(coin_change_30d, 2),
        'vs_btc_pct': round(coin_change_30d - btc_change_30d, 2),
        'scores': {
            'momentum': round(momentum, 1),
            'trend': round(trend, 1),
            'volume': round(volume, 1),
            'relative': round(relative, 1),
            'consistency': round(consistency, 1),
            'btc_bonus': BTC_SCORE_BONUS if symbol == 'BTCUSDT' else 0,
        },
        'total_score': round(total, 1),
    }


def rank_universe(client):
    """Score every pair in the universe, return sorted list (best first).
    This is the main entry point called by the background thread."""
    log.info(f"Ranker: scoring {len(UNIVERSE)} pairs")
    start = time.time()

    # Get BTC's 30d change first - needed as baseline for relative scoring
    btc_klines = _fetch_klines(client, 'BTCUSDT', interval='1d', limit=60)
    if not btc_klines or len(btc_klines) < 30:
        log.warning("Ranker: BTC klines unavailable, using 0 as baseline")
        btc_change_30d = 0.0
    else:
        btc_closes = [float(k[4]) for k in btc_klines]
        btc_change_30d = _pct(btc_closes[-30], btc_closes[-1])

    results = []
    for symbol in UNIVERSE:
        try:
            r = score_pair(client, symbol, btc_change_30d)
            if r:
                results.append(r)
        except Exception as e:
            log.debug(f"Score failed for {symbol}: {e}")

    results.sort(key=lambda r: r['total_score'], reverse=True)
    elapsed = time.time() - start
    log.info(f"Ranker: scored {len(results)}/{len(UNIVERSE)} pairs in {elapsed:.1f}s. "
             f"Top 3: {[r['symbol'] for r in results[:3]]}")
    return results


def save_rankings(rankings, meta=None):
    """Persist rankings to disk for the dashboard to read."""
    try:
        os.makedirs(os.path.dirname(_RANKING_PATH), exist_ok=True)
        payload = {
            'updated_at': datetime.utcnow().isoformat() + 'Z',
            'universe_size': len(UNIVERSE),
            'scored_count': len(rankings),
            'rankings': rankings,
        }
        if meta:
            payload.update(meta)
        with open(_RANKING_PATH, 'w') as f:
            json.dump(payload, f, indent=2)
        log.info(f"Ranker: saved rankings to {_RANKING_PATH}")
    except Exception as e:
        log.warning(f"Ranker: failed to save rankings: {e}")


def load_rankings():
    """Read the last saved rankings. Called by the dashboard endpoint."""
    try:
        if not os.path.exists(_RANKING_PATH):
            return None
        with open(_RANKING_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.debug(f"Ranker: read failed: {e}")
        return None


class RankerThread:
    """Background thread that runs the ranker on a fixed interval.
    Started once from app.py after trader is initialized."""

    def __init__(self, trader):
        self.trader = trader
        self._stop = False
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name='ranker')
        self._thread.start()
        log.info("Ranker thread started")

    def stop(self):
        self._stop = True

    def _run(self):
        # Do an initial run immediately so the dashboard has data on first load
        try:
            rankings = rank_universe(self.trader.client)
            save_rankings(rankings)
        except Exception as e:
            log.warning(f"Ranker initial run failed: {e}")

        while not self._stop:
            time.sleep(_RANKING_INTERVAL_SECONDS)
            if self._stop:
                break
            try:
                rankings = rank_universe(self.trader.client)
                save_rankings(rankings)
            except Exception as e:
                log.warning(f"Ranker cycle failed: {e}")


def run_once(trader):
    """Trigger a ranking run right now, on-demand. Called by API refresh."""
    rankings = rank_universe(trader.client)
    save_rankings(rankings)
    return rankings
