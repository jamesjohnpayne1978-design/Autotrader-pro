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
#
# Reduced from 15 -> 10 after the anti-froth rewrite because momentum weights
# were rebalanced away from raw price change (which BTC often loses on).
BTC_SCORE_BONUS = 10.0

# Blacklist - pairs that never rank, no matter how they score. Prevents
# the ranker from suggesting coins the user has removed for cause (bad
# fill quality, bad win rate history, delisted, tokenized stocks, etc.)
# Extend this list from user settings later if we want it configurable.
BLACKLIST = {'NEARUSDT', 'INJUSDT'}

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


def _compute_rsi(closes, period=14):
    """Standard Wilder RSI on a series of closes. Returns 50 if not enough
    data (neutral - won't trigger any penalty or bonus)."""
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
    # Use last `period` values
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _score_momentum(price_30d_ago, price_now):
    """0-25 points (REDUCED from 40). Uses 30d price change, sqrt-scaled
    to reward strong trends without letting outliers dominate. Also HARD
    CAPPED so a parabolic 200% pump doesn't get max score. Beyond 60%
    gain, additional gain gives diminishing returns - and by 100%+ we
    apply a soft penalty (see _score_reversion_penalty below)."""
    change = _pct(price_30d_ago, price_now)
    if change <= 0:
        return 0.0
    # Sqrt scale: 25% -> ~12, 50% -> ~18, 100%+ -> 25 (capped)
    scaled = (change ** 0.5) * 2.5
    return min(25.0, scaled)


def _score_long_term_trend(closes):
    """0-15 points (NEW). Rewards coins with steady multi-week uptrends,
    not just parabolic short-term pumps. Uses 60d/90d change if we have
    the history. Coins with strong 90d trends are much less likely to
    mean-revert than coins with only strong 30d trends."""
    if len(closes) < 60:
        return 0.0
    change_60d = _pct(closes[-60], closes[-1])
    if change_60d <= 0:
        return 0.0
    # 20% over 60d -> ~7 pts, 50% -> ~12, 100%+ -> 15 (capped)
    return min(15.0, (change_60d ** 0.5) * 1.5)


def _score_trend(closes, ma_period=50):
    """0-20 points (REDUCED from 25). Distance above MA + slope of MA."""
    if len(closes) < ma_period + 5:
        return 0.0
    ma_now = sum(closes[-ma_period:]) / ma_period
    ma_prev = sum(closes[-ma_period - 5:-5]) / ma_period
    price = closes[-1]
    if ma_now <= 0:
        return 0.0
    # Points for price above MA (0-12)
    distance_pct = _pct(ma_now, price)
    distance_score = min(12.0, max(0.0, distance_pct * 0.6))
    # Points for MA rising (0-8)
    slope_pct = _pct(ma_prev, ma_now)
    slope_score = min(8.0, max(0.0, slope_pct * 1.6))
    return distance_score + slope_score


def _score_volume(volumes):
    """0-10 points (REDUCED from 15). Recent 7d avg volume vs 30d avg."""
    if len(volumes) < 30:
        return 0.0
    vol_7d = sum(volumes[-7:]) / 7
    vol_30d = sum(volumes[-30:]) / 30
    if vol_30d <= 0:
        return 0.0
    ratio = vol_7d / vol_30d
    if ratio < 1.0:
        return 0.0
    return min(10.0, (ratio - 1.0) * 10.0)


def _score_relative_to_btc(coin_change_30d, btc_change_30d):
    """0-10 points (REDUCED from 15). Outperformance vs BTC over 30d."""
    if coin_change_30d <= btc_change_30d:
        return 0.0
    outperformance = coin_change_30d - btc_change_30d
    return min(10.0, outperformance * 0.33)


def _score_consistency(closes):
    """0-20 points (INCREASED from 5). This is now MUCH more important.
    Sharpe-like: mean daily return / stdev. Rewards smooth uptrends over
    parabolic pumps with the same net gain. A coin that went up 50% in
    a straight line scores much higher than one that went 0-0-0-0-50 in
    one day. Parabolic moves reverse. Steady moves persist."""
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
    # 0.05 -> 5, 0.1 -> 10, 0.2+ -> 20
    return max(0.0, min(20.0, sharpe_like * 100))


def _overheat_penalty(rsi_14d):
    """0 to -30 points. Applies a growing penalty when daily RSI enters
    overbought territory. This is the KEY anti-froth guardrail - a coin
    at RSI 85 is statistically very likely to correct within 2 weeks.

    RSI < 65: no penalty
    RSI 65-70: -5 (early warning)
    RSI 70-75: -12 (overbought)
    RSI 75-80: -20 (extended)
    RSI 80+: -30 (extreme, likely reversion soon)
    """
    if rsi_14d < 65:
        return 0.0
    if rsi_14d < 70:
        return -5.0
    if rsi_14d < 75:
        return -12.0
    if rsi_14d < 80:
        return -20.0
    return -30.0


def _parabolic_penalty(closes):
    """0 to -25 points. Detects coins where most of the 30d gain came
    in the last 7 days - classic parabolic profile that mean-reverts.

    Ratio = 7d_change / 30d_change:
      < 0.35: healthy distribution, no penalty
      0.35-0.55: -5
      0.55-0.75: -15
      > 0.75: -25 (nearly all gains in last week = extreme parabolic)

    Also penalizes if 7d change alone > 40% (regardless of 30d) since
    that's parabolic on its own.
    """
    if len(closes) < 30:
        return 0.0
    change_7d = _pct(closes[-8], closes[-1])
    change_30d = _pct(closes[-30], closes[-1])

    # Only apply if there IS a positive 30d trend to distribute
    if change_30d <= 5:
        return 0.0

    # Standalone parabolic: 7d gain > 40%
    if change_7d > 40:
        return -25.0

    # Ratio-based
    ratio = change_7d / change_30d if change_30d > 0 else 0
    if ratio < 0.35:
        return 0.0
    if ratio < 0.55:
        return -5.0
    if ratio < 0.75:
        return -15.0
    return -25.0


def _reversion_penalty(change_30d):
    """0 to -20 points. Direct penalty for coins that have already run
    hard - the higher the 30d gain, the more likely near-term reversion.
    Complements the parabolic penalty (which looks at distribution) with
    a raw magnitude check.

    30d change < 50%: no penalty (still normal upside)
    50-80%: -5
    80-120%: -12
    >120%: -20 (definitely due to correct)
    """
    if change_30d < 50:
        return 0.0
    if change_30d < 80:
        return -5.0
    if change_30d < 120:
        return -12.0
    return -20.0


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

    Post anti-froth rewrite: scoring is now a mix of POSITIVE factors
    (max ~100) and NEGATIVE penalties (max ~-75). A parabolic pump like
    ARB +106% will score:
      + momentum 25 (capped) + relative 10 (capped) + volume ~8 = ~43
      - reversion 20 (>120%) - parabolic 25 - overheat 20 (RSI 80+) = -65
      Net: ~-22 -> falls out of top 10, correctly avoided.

    Meanwhile BTC +25% steady trend scores:
      + momentum ~12 + long-term 12 + trend 15 + consistency 15 +
        relative 0 + volume 5 + BTC bonus 10 = ~69
      - no penalties
      Net: 69 -> stays near the top, correctly rewarded.
    """
    # Blacklist check first - skip entirely, don't waste API calls
    if symbol in BLACKLIST:
        return None

    # Fetch 90d of data so we can compute the long-term trend factor
    klines = _fetch_klines(client, symbol, interval='1d', limit=95)
    if not klines or len(klines) < 30:
        return None
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    price_now = closes[-1]
    price_30d_ago = closes[-30]
    coin_change_30d = _pct(price_30d_ago, price_now)

    # Positive scoring factors (max ~100)
    momentum = _score_momentum(price_30d_ago, price_now)          # 0-25
    long_term = _score_long_term_trend(closes)                    # 0-15
    trend = _score_trend(closes)                                  # 0-20
    volume = _score_volume(volumes)                               # 0-10
    relative = _score_relative_to_btc(coin_change_30d, btc_change_30d)  # 0-10
    consistency = _score_consistency(closes)                      # 0-20

    positive = momentum + long_term + trend + volume + relative + consistency

    # Penalty factors (max ~-75)
    rsi_14d = _compute_rsi(closes, period=14)
    overheat = _overheat_penalty(rsi_14d)              # 0 to -30
    parabolic = _parabolic_penalty(closes)             # 0 to -25
    reversion = _reversion_penalty(coin_change_30d)    # 0 to -20

    penalties = overheat + parabolic + reversion

    # BTC bonus - keeps BTC competitive in stable markets
    btc_bonus = BTC_SCORE_BONUS if symbol == 'BTCUSDT' else 0.0

    total = positive + penalties + btc_bonus
    # Never let score go negative (dashboard rendering assumes 0-100 range)
    total = max(0.0, total)

    return {
        'symbol': symbol,
        'display': symbol.replace('USDT', '/USDT'),
        'price': price_now,
        'change_30d_pct': round(coin_change_30d, 2),
        'vs_btc_pct': round(coin_change_30d - btc_change_30d, 2),
        'rsi_14d': round(rsi_14d, 1),
        'scores': {
            'momentum': round(momentum, 1),
            'long_term': round(long_term, 1),
            'trend': round(trend, 1),
            'volume': round(volume, 1),
            'relative': round(relative, 1),
            'consistency': round(consistency, 1),
            'overheat_penalty': round(overheat, 1),
            'parabolic_penalty': round(parabolic, 1),
            'reversion_penalty': round(reversion, 1),
            'btc_bonus': btc_bonus,
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
