"""
winners_allocator.py - Concentration allocator with optional ranker-driven targets.

Two modes (both only act when concentration_mode_enabled is True):

FIXED (default)
    SUI 40% / BNB 25% / BTC 20% / USDT 15%  - the original winners-by-win-rate
    split. Unchanged behaviour.

AUTO TARGETS (toggle: /api/concentration/auto-targets)
    BTC stays a permanent 20% anchor. The two other slots (40% and 25%) are
    filled by the momentum ranker (ranker.py) instead of being hard-coded.
    The ranker only ever *proposes*; these rules decide whether to act:

      * Seeding      - turning it on starts from the current fixed picks
                       (SUI / BNB), so enabling it trades nothing by itself.
      * Once/ranking - each ranker snapshot is evaluated once (not once per
                       hourly wake-up), so confirmation counts are real.
      * Hysteresis   - a challenger must beat the weakest held coin by
                       AUTO_SWAP_MARGIN points on AUTO_CONFIRM_RANKINGS
                       consecutive rankings before it can replace it.
      * Min hold     - a coin must have been held AUTO_MIN_HOLD_HOURS before it
                       can be voluntarily swapped out.
      * One change   - at most one slot changes per ranking.
      * Entry filter - challengers need a minimum score, RSI below a cap and no
                       parabolic-pump penalty (anti-froth, on top of the ranker's).
      * Exit floor   - a held coin scoring below AUTO_EXIT_SCORE for
                       AUTO_CONFIRM_RANKINGS rankings is dropped (min-hold does
                       not apply). If nothing eligible replaces it, the slot
                       sits in cash until something qualifies.
      * Stale data   - if the ranking file is older than AUTO_STALE_HOURS the
                       current picks are frozen (no rotation, no forced exit).
      * Slot-tied weights - a swap hands the leaving coin's weight to the new
                       coin; picks are never reshuffled between the 40%/25%
                       slots (that would be pure fee churn).

Rebalance triggers (unchanged):
    - Any target drifts >5% (absolute) from its target percentage
    - OR 7 days elapsed since the last rebalance
    - AND at least 6 hours since the previous rebalance (rate limit)
    - NEW: a rotation that just changed targets rebalances immediately
      (bypasses the 6h rate limit once).

Rebalance mechanics (unchanged):
    1. Compute portfolio value + current allocation (from Binance balances)
    2. Cancel open OCOs on target pairs AND on any pair about to be sold
    3. SELL non-target holdings (100%) and trim over-target pairs
    4. BUY under-target pairs
    5. Cancel the auto-OCOs the buys created (positions hold; rebalance sizes)
    6. Telegram preview before, summary after

Safety:
    - Every trade goes through trader.execute_trade with bypass_cooldown=True
    - Individual trade failures do NOT abort the whole rebalance
    - Telegram preview is sent BEFORE execution
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta

try:
    import ranker as _ranker
except Exception:  # pragma: no cover - ranker missing must not break the bot
    _ranker = None

log = logging.getLogger(__name__)

_STATE_PATH = '/data/concentration_state.json'
_PAIRS_OVERRIDE_PATH = '/data/trading_pairs.json'   # same file app.py uses
_MIN_TRADE_USDT = 10.0            # Skip rebalance moves below this ($)
_MIN_INTERVAL_SECONDS = 6 * 3600   # Rate limit: 6h between rebalances
_MAX_INTERVAL_SECONDS = 7 * 86400  # Force rebalance if 7 days elapsed
_CHECK_INTERVAL_SECONDS = 3600     # How often the scheduler wakes to check
_STABLECOINS = {'USDT', 'BUSD', 'USDC', 'FDUSD', 'TUSD', 'DAI'}

_STATE_LOCK = threading.RLock()


# Fixed target allocations - fraction of total portfolio value.
# Sum of these + implied cash = 1.0. Cash is whatever's left.
DEFAULT_TARGETS = {
    'BTCUSDT': 0.20,
    'BNBUSDT': 0.25,
    'SUIUSDT': 0.40,
    # USDT implied: 1.0 - 0.85 = 0.15
}

# Drift threshold: rebalance if any target pair is off by more than this
# (as absolute percentage points, not relative)
DEFAULT_DRIFT_THRESHOLD_PCT = 5.0

# ---------------------------------------------------------------------------
# Auto-targets tuning knobs. All in one place so they're easy to adjust.
# ---------------------------------------------------------------------------
AUTO_ANCHOR = 'BTCUSDT'            # permanent anchor, never rotated out
AUTO_ANCHOR_WEIGHT = 0.20
AUTO_SLOT_WEIGHTS = [0.40, 0.25]   # rotating slots, in order
AUTO_ENTRY_MIN_SCORE = 45.0        # challenger needs at least this ranker score
AUTO_EXIT_SCORE = 30.0             # held coin below this (confirmed) is dropped
AUTO_SWAP_MARGIN = 8.0             # challenger must beat weakest held by this
AUTO_CONFIRM_RANKINGS = 2          # consecutive rankings required (4h apart)
AUTO_MIN_HOLD_HOURS = 48.0         # min hold before a voluntary swap
AUTO_STALE_HOURS = 8.0             # ranking older than this => freeze picks
AUTO_ENTRY_MAX_RSI = 72.0          # don't enter coins at/above this daily RSI
AUTO_ENTRY_MAX_PARABOLIC = -15.0   # parabolic_penalty must be ABOVE this


def _utcnow():
    return datetime.utcnow()


def _iso(dt):
    return dt.isoformat() + 'Z'


def _parse_iso(s):
    try:
        return datetime.fromisoformat(str(s).replace('Z', ''))
    except Exception:
        return None


def _short(sym):
    return str(sym).replace('USDT', '') if sym else '-'


def _load_state():
    """Persist last rebalance time + history across restarts."""
    try:
        if os.path.exists(_STATE_PATH):
            with open(_STATE_PATH) as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _save_state(state):
    with _STATE_LOCK:
        try:
            os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
            with open(_STATE_PATH, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.warning(f"Concentration state save failed: {e}")


class WinnersAllocator:
    """Encapsulates the concentration rebalance logic. One instance created
    at bot startup, background thread calls maybe_rebalance() hourly."""

    def __init__(self, trader, config, signal_engine=None):
        self.trader = trader
        self.config = config
        self.signal_engine = signal_engine  # used for _tg_send if available
        self.targets = dict(DEFAULT_TARGETS)
        self.drift_threshold_pct = DEFAULT_DRIFT_THRESHOLD_PCT
        self._stop = False
        self._thread = None
        self._lock = threading.RLock()
        self._state = _load_state()
        self._refresh_targets()

    # ---------- Enable / disable ----------

    def is_enabled(self):
        """Read the toggle from config. Kept as a method so it's re-checked
        every cycle - user can toggle mid-run and the next cycle picks it up."""
        return bool(getattr(self.config, 'concentration_mode_enabled', False))

    def auto_enabled(self):
        """Ranker-driven targets toggle. Stored in the allocator's own state
        file so it survives restarts without touching app.py settings."""
        return bool(self._state.get('auto_targets_enabled', False))

    def set_auto_enabled(self, enabled):
        """Flip auto targets on/off. ON seeds from the current fixed picks
        (so nothing trades by itself). OFF discards auto state and returns to
        the fixed split."""
        enabled = bool(enabled)
        with self._lock:
            self._state['auto_targets_enabled'] = enabled
            if not enabled:
                self._state.pop('auto', None)
            _save_state(self._state)
            self._refresh_targets()
        log.info(f"Concentration auto targets {'ENABLED' if enabled else 'DISABLED'}")
        return enabled

    def target_symbols(self):
        """Current target symbols (for syncing trading_pairs)."""
        self._refresh_targets()
        return list(self.targets.keys())

    def _ensure_baseline(self, total_value):
        """First time concentration mode is enabled, capture the current
        portfolio value as the baseline. All subsequent performance tracking
        measures against this. If user toggles OFF then ON again, we keep
        the ORIGINAL baseline so the metric represents the full concentration
        experiment, not just the latest session.

        Passing total_value in so we don't fetch portfolio twice per call.
        """
        if self._state.get('baseline_value') is not None:
            return
        if total_value <= 0:
            return
        self._state['baseline_value'] = round(total_value, 2)
        self._state['baseline_at'] = _iso(_utcnow())
        _save_state(self._state)
        log.info(f"Concentration baseline captured: ${total_value:.2f} at "
                 f"{self._state['baseline_at']}")

    def _compute_performance(self, total_value):
        """Return performance stats vs the baseline captured when
        concentration was first enabled. Returns None if no baseline set."""
        baseline = self._state.get('baseline_value')
        baseline_at = self._state.get('baseline_at')
        if not baseline or baseline <= 0 or not baseline_at:
            return None
        try:
            dt = datetime.fromisoformat(baseline_at.replace('Z', ''))
            hours_elapsed = (_utcnow() - dt).total_seconds() / 3600.0
        except Exception:
            hours_elapsed = 0
        change_usdt = total_value - baseline
        change_pct = (change_usdt / baseline) * 100 if baseline > 0 else 0
        return {
            'baseline_value': baseline,
            'baseline_at': baseline_at,
            'current_value': round(total_value, 2),
            'change_usdt': round(change_usdt, 2),
            'change_pct': round(change_pct, 2),
            'hours_elapsed': round(hours_elapsed, 1),
            'days_elapsed': round(hours_elapsed / 24, 1),
        }

    # =====================================================================
    # AUTO TARGETS
    # =====================================================================

    def _ensure_auto_seeded(self):
        """Return the auto-state dict, creating it (seeded from the fixed
        split's non-anchor coins, biggest weight first) if missing."""
        auto = self._state.get('auto')
        if isinstance(auto, dict) and isinstance(auto.get('slots'), list) and auto['slots']:
            return auto
        seeds = sorted(((s, w) for s, w in DEFAULT_TARGETS.items() if s != AUTO_ANCHOR),
                       key=lambda x: -x[1])
        now = _iso(_utcnow())
        slots = []
        for i, weight in enumerate(AUTO_SLOT_WEIGHTS):
            slots.append({
                'symbol': seeds[i][0] if i < len(seeds) else None,
                'weight': weight,
                'since': now,
            })
        auto = {
            'slots': slots,
            'streaks': {},
            'low_counts': {},
            'history': [],
            'last_ranking_at': None,
            'pending_rebalance': False,
            'stale_notified': False,
            'started_at': now,
        }
        self._state['auto'] = auto
        _save_state(self._state)
        log.info(f"Auto targets seeded: {[s['symbol'] for s in slots]}")
        return auto

    @staticmethod
    def _targets_from_slots(slots):
        targets = {AUTO_ANCHOR: AUTO_ANCHOR_WEIGHT}
        for s in slots:
            if s.get('symbol'):
                targets[s['symbol']] = s['weight']
        return targets

    def _refresh_targets(self):
        """Recompute self.targets from persisted state. Cheap; called often."""
        with self._lock:
            if not self.auto_enabled():
                self.targets = dict(DEFAULT_TARGETS)
                return self.targets
            auto = self._ensure_auto_seeded()
            self.targets = self._targets_from_slots(auto['slots'])
            return self.targets

    @staticmethod
    def _eligible_entry(r):
        """Anti-froth gate for NEW picks (held coins are judged by the exit
        floor instead)."""
        try:
            scores = r.get('scores') or {}
            return (
                float(r.get('total_score', 0)) >= AUTO_ENTRY_MIN_SCORE
                and float(r.get('rsi_14d', 50)) < AUTO_ENTRY_MAX_RSI
                and float(scores.get('parabolic_penalty', 0)) > AUTO_ENTRY_MAX_PARABOLIC
            )
        except Exception:
            return False

    @staticmethod
    def _ranking_age_hours(ts):
        dt = _parse_iso(ts) if ts else None
        if not dt:
            return None
        return (_utcnow() - dt).total_seconds() / 3600.0

    @staticmethod
    def _hours_since(ts):
        dt = _parse_iso(ts) if ts else None
        if not dt:
            return 0.0
        return max(0.0, (_utcnow() - dt).total_seconds() / 3600.0)

    def _apply_slot_change(self, auto, slot, best, kind, by_sym):
        """Mutate a slot in place and record the event."""
        out_sym = slot.get('symbol')
        in_sym = best['symbol'] if best else None
        now = _utcnow()
        slot['symbol'] = in_sym
        slot['since'] = _iso(now)
        auto.setdefault('low_counts', {}).pop(out_sym, None)
        auto['streaks'] = {}
        event = {
            'at': _iso(now),
            'type': kind,                      # swap | forced_exit | fill
            'slot_weight_pct': round(slot['weight'] * 100, 1),
            'out': out_sym,
            'in': in_sym,
            'out_score': by_sym.get(out_sym, {}).get('total_score') if out_sym else None,
            'in_score': best.get('total_score') if best else None,
        }
        hist = auto.setdefault('history', [])
        hist.append(event)
        auto['history'] = hist[-20:]
        return event

    def _process_ranking(self, auto, rankings):
        """Evaluate one ranker snapshot against the current picks. Mutates
        `auto`; returns a list with at most one event."""
        by_sym = {r['symbol']: r for r in rankings if r.get('symbol')}
        slots = auto['slots']
        held = {s['symbol'] for s in slots if s.get('symbol')}
        streaks_old = auto.get('streaks', {}) or {}
        low = auto.setdefault('low_counts', {})

        # Consecutive-rankings-below-exit-floor counters for held coins.
        # A coin missing from this ranking (data hiccup) is left untouched.
        for s in slots:
            sym = s.get('symbol')
            r = by_sym.get(sym) if sym else None
            if r is None:
                continue
            if float(r['total_score']) < AUTO_EXIT_SCORE:
                low[sym] = low.get(sym, 0) + 1
            else:
                low[sym] = 0
        for k in list(low):
            if k not in held:
                del low[k]

        challengers = sorted(
            (r for r in rankings
             if r.get('symbol') and r['symbol'] != AUTO_ANCHOR
             and r['symbol'] not in held and self._eligible_entry(r)),
            key=lambda r: r['total_score'], reverse=True)
        best = challengers[0] if challengers else None

        event = None
        new_streaks = {}

        # 1) Forced exit: confirmed below the floor. Lowest score goes first.
        forced = [s for s in slots
                  if s.get('symbol') in by_sym and low.get(s['symbol'], 0) >= AUTO_CONFIRM_RANKINGS]
        if forced:
            slot = min(forced, key=lambda s: by_sym[s['symbol']]['total_score'])
            event = self._apply_slot_change(auto, slot, best, 'forced_exit', by_sym)
        else:
            empty = next((s for s in slots if not s.get('symbol')), None)
            if empty is not None:
                # 2) Fill an empty slot (left in cash by an earlier forced exit)
                if best:
                    n = streaks_old.get(best['symbol'], 0) + 1
                    if n >= AUTO_CONFIRM_RANKINGS:
                        event = self._apply_slot_change(auto, empty, best, 'fill', by_sym)
                    else:
                        new_streaks[best['symbol']] = n
            elif best:
                # 3) Voluntary swap: best challenger vs weakest known held coin
                known = [s for s in slots if s.get('symbol') in by_sym]
                if known:
                    weakest = min(known, key=lambda s: by_sym[s['symbol']]['total_score'])
                    margin = float(best['total_score']) - float(by_sym[weakest['symbol']]['total_score'])
                    if margin >= AUTO_SWAP_MARGIN:
                        n = streaks_old.get(best['symbol'], 0) + 1
                        held_h = self._hours_since(weakest.get('since'))
                        if n >= AUTO_CONFIRM_RANKINGS and held_h >= AUTO_MIN_HOLD_HOURS:
                            event = self._apply_slot_change(auto, weakest, best, 'swap', by_sym)
                        else:
                            new_streaks[best['symbol']] = n

        if event is None:
            auto['streaks'] = new_streaks
            return []
        auto['pending_rebalance'] = True
        return [event]

    def _update_auto_targets(self):
        """Evaluate the latest ranker snapshot (once per snapshot). Persists
        state, refreshes targets, syncs trading_pairs and sends Telegram if a
        slot changed. Returns the list of change events."""
        if not self.auto_enabled() or _ranker is None:
            return []
        stale_msg = None
        events = []
        with self._lock:
            auto = self._ensure_auto_seeded()
            data = _ranker.load_rankings()
            if not data or not data.get('rankings'):
                return []
            ts = data.get('updated_at')
            age_h = self._ranking_age_hours(ts)
            if age_h is None or age_h > AUTO_STALE_HOURS:
                # Freeze: never rotate or force-exit on stale data
                if not auto.get('stale_notified'):
                    auto['stale_notified'] = True
                    _save_state(self._state)
                    age_txt = 'unknown age' if age_h is None else f"{age_h:.1f}h old"
                    stale_msg = (f"⚠️ *Auto targets frozen* - ranker data is {age_txt}. "
                                 f"Keeping current picks until it refreshes.")
                events = []
            else:
                if auto.get('stale_notified'):
                    auto['stale_notified'] = False
                if ts == auto.get('last_ranking_at'):
                    _save_state(self._state)
                    return []
                events = self._process_ranking(auto, data['rankings'])
                auto['last_ranking_at'] = ts
                self.targets = self._targets_from_slots(auto['slots'])
                _save_state(self._state)

        if stale_msg:
            try:
                self._tg_send(stale_msg, context='conc-auto-stale')
            except Exception:
                pass
        for ev in events:
            log.info(f"Auto targets change: {ev}")
            try:
                self._tg_rotation(ev)
            except Exception as e:
                log.debug(f"Rotation telegram failed: {e}")
        return events

    def _auto_status(self):
        """Dashboard block describing the auto-target state."""
        auto = self._state.get('auto') or {}
        data = None
        try:
            data = _ranker.load_rankings() if _ranker else None
        except Exception:
            data = None
        rankings = (data or {}).get('rankings') or []
        by_sym = {r['symbol']: r for r in rankings if r.get('symbol')}
        rank_pos = {r['symbol']: i + 1 for i, r in enumerate(rankings) if r.get('symbol')}
        age_h = self._ranking_age_hours((data or {}).get('updated_at'))

        slots_out = []
        for s in auto.get('slots', []):
            sym = s.get('symbol')
            r = by_sym.get(sym) if sym else None
            held_h = self._hours_since(s.get('since')) if sym else 0.0
            slots_out.append({
                'symbol': sym,
                'asset': _short(sym) if sym else None,
                'weight_pct': round(s.get('weight', 0) * 100, 1),
                'held_hours': round(held_h, 1),
                'min_hold_remaining_hours': round(max(0.0, AUTO_MIN_HOLD_HOURS - held_h), 1) if sym else 0,
                'score': r.get('total_score') if r else None,
                'rank': rank_pos.get(sym),
                'rsi_14d': r.get('rsi_14d') if r else None,
                'below_exit_floor_count': (auto.get('low_counts') or {}).get(sym, 0) if sym else 0,
            })
        return {
            'anchor': AUTO_ANCHOR,
            'anchor_weight_pct': round(AUTO_ANCHOR_WEIGHT * 100, 1),
            'slots': slots_out,
            'challenger_streaks': auto.get('streaks', {}),
            'ranking_updated_at': (data or {}).get('updated_at'),
            'ranking_age_hours': round(age_h, 1) if age_h is not None else None,
            'ranking_stale': (age_h is None) or (age_h > AUTO_STALE_HOURS),
            'pending_rebalance': bool(auto.get('pending_rebalance')),
            'recent_changes': (auto.get('history') or [])[-5:],
            'rules': {
                'entry_min_score': AUTO_ENTRY_MIN_SCORE,
                'exit_score': AUTO_EXIT_SCORE,
                'swap_margin': AUTO_SWAP_MARGIN,
                'confirm_rankings': AUTO_CONFIRM_RANKINGS,
                'min_hold_hours': AUTO_MIN_HOLD_HOURS,
                'stale_hours': AUTO_STALE_HOURS,
                'entry_max_rsi': AUTO_ENTRY_MAX_RSI,
            },
        }

    def _sync_trading_pairs(self):
        """Keep config.trading_pairs (and its on-disk override) equal to the
        current targets while concentration mode is on, so rotated-in coins
        are known to the rest of the bot. No-op when already in sync."""
        try:
            if not self.is_enabled():
                return
            targets = list(self.targets.keys())
            current = list(getattr(self.config, 'trading_pairs', []) or [])
            if set(current) == set(targets):
                return
            self.config.trading_pairs = targets
            os.makedirs(os.path.dirname(_PAIRS_OVERRIDE_PATH), exist_ok=True)
            with open(_PAIRS_OVERRIDE_PATH, 'w') as f:
                json.dump(targets, f)
            log.info(f"Concentration: trading_pairs synced to targets {targets}")
        except Exception as e:
            log.warning(f"Concentration pairs sync failed: {e}")

    # ---------- State inspection (for API) ----------

    def get_status(self):
        """Return a snapshot for /api/concentration/status - dashboard reads
        this to show current vs target allocation, drift, and history."""
        self._refresh_targets()
        targets = dict(self.targets)
        enabled = self.is_enabled()
        try:
            portfolio = self.trader.get_portfolio()
            total_value = float(portfolio.get('total_usdt', 0) or 0)
        except Exception as e:
            log.debug(f"Concentration get_portfolio failed: {e}")
            total_value = 0.0

        current = self._compute_current_allocation(total_value)

        # Capture baseline on first call when concentration is enabled.
        # (Called from get_status() which is polled frequently, so we get
        # this within seconds of the mode being turned on.)
        if enabled and total_value > 0:
            self._ensure_baseline(total_value)

        performance = self._compute_performance(total_value) if enabled else None

        rows = []
        max_drift = 0.0
        for sym, tgt_pct in targets.items():
            base = sym.replace('USDT', '')
            cur_val = current.get(base, 0.0)
            cur_pct = (cur_val / total_value * 100) if total_value > 0 else 0.0
            drift = cur_pct - (tgt_pct * 100)
            max_drift = max(max_drift, abs(drift))
            rows.append({
                'symbol': sym,
                'asset': base,
                'target_pct': round(tgt_pct * 100, 1),
                'current_pct': round(cur_pct, 1),
                'current_value_usdt': round(cur_val, 2),
                'target_value_usdt': round(tgt_pct * total_value, 2),
                'drift_pct': round(drift, 1),
            })
        # Add implied cash target
        cash_target_pct = (1.0 - sum(targets.values())) * 100
        cash_current = current.get('USDT', 0.0)
        cash_current_pct = (cash_current / total_value * 100) if total_value > 0 else 0.0
        cash_drift = cash_current_pct - cash_target_pct
        max_drift = max(max_drift, abs(cash_drift))
        rows.append({
            'symbol': 'USDT',
            'asset': 'USDT',
            'target_pct': round(cash_target_pct, 1),
            'current_pct': round(cash_current_pct, 1),
            'current_value_usdt': round(cash_current, 2),
            'target_value_usdt': round((cash_target_pct / 100) * total_value, 2),
            'drift_pct': round(cash_drift, 1),
        })

        # List non-target assets currently held (would be sold next rebalance)
        non_target = []
        target_bases = {s.replace('USDT', '') for s in targets}
        for asset, val in current.items():
            if asset in target_bases or asset in _STABLECOINS:
                continue
            if val >= _MIN_TRADE_USDT:
                non_target.append({'asset': asset, 'value_usdt': round(val, 2)})

        last_rebalance = self._state.get('last_rebalance_at')
        history = self._state.get('history', [])[-10:]  # last 10 events

        return {
            'enabled': enabled,
            'mode': 'auto' if self.auto_enabled() else 'fixed',
            'total_portfolio_value': round(total_value, 2),
            'drift_threshold_pct': self.drift_threshold_pct,
            'max_drift_pct': round(max_drift, 1),
            'needs_rebalance': max_drift > self.drift_threshold_pct,
            'targets': rows,
            'non_target_holdings': non_target,
            'last_rebalance_at': last_rebalance,
            'next_scheduled_check_hours': self._hours_until_next_forced_rebalance(),
            'history': history,
            'performance': performance,
            'auto': self._auto_status() if self.auto_enabled() else None,
        }

    def _hours_until_next_forced_rebalance(self):
        """When will the 7-day timer force a rebalance regardless of drift?"""
        last = self._state.get('last_rebalance_at')
        if not last:
            return 0
        try:
            last_dt = datetime.fromisoformat(last.replace('Z', ''))
            elapsed = (_utcnow() - last_dt).total_seconds()
            remaining = _MAX_INTERVAL_SECONDS - elapsed
            return max(0, round(remaining / 3600, 1))
        except Exception:
            return 0

    def _compute_current_allocation(self, total_value):
        """Fetch current holdings from Binance, value them in USDT.
        Returns {asset: usdt_value}."""
        result = {}
        try:
            acct = self.trader.client.get_account()
            prices = None
            for bal in acct['balances']:
                asset = bal['asset']
                total = float(bal['free']) + float(bal['locked'])
                if total <= 0:
                    continue
                if asset in _STABLECOINS:
                    result[asset] = result.get(asset, 0) + total
                    continue
                # Value in USDT
                if prices is None:
                    try:
                        prices = {p['symbol']: float(p['price']) for p in self.trader.client.get_all_tickers()}
                    except Exception:
                        prices = {}
                sym = asset + 'USDT'
                if sym in prices:
                    val = total * prices[sym]
                    if val >= 1.0:  # ignore dust below $1
                        result[asset] = val
        except Exception as e:
            log.warning(f"Concentration current alloc fetch failed: {e}")
        return result

    # ---------- OCO management ----------

    def _cancel_target_ocos(self, extra_symbols=()):
        """Cancel any open orders on target pairs (plus any extra symbols,
        e.g. coins about to be sold). Concentration mode holds positions
        long-term - it exits/sizes via rebalance drift, not via TP/SL. Any
        OCO on target coins would fire at 6% and force a costly re-entry loop
        (sell -> drift alert -> rebuy at market). Removing them lets positions
        ride freely; rebalance handles all sizing.

        Called BEFORE rebalance (to free locked balances for selling) and
        AFTER (to remove the new OCOs that trader.execute_trade auto-places
        on every buy).
        """
        cancelled = 0
        symbols = list(dict.fromkeys(list(self.targets.keys()) + list(extra_symbols)))
        for sym in symbols:
            try:
                open_orders = self.trader.client.get_open_orders(symbol=sym)
                for order in open_orders:
                    try:
                        self.trader.client.cancel_order(
                            symbol=sym,
                            orderId=order['orderId']
                        )
                        cancelled += 1
                        log.info(f"Concentration: cancelled order {order['orderId']} on {sym}")
                    except Exception as e:
                        log.debug(f"Cancel order failed for {sym}/{order.get('orderId')}: {e}")
            except Exception as e:
                log.debug(f"Get open orders failed for {sym}: {e}")
        if cancelled:
            log.info(f"Concentration: cancelled {cancelled} open OCO orders")
        return cancelled

    # ---------- Rebalance decision ----------

    def _can_rebalance_now(self):
        """Rate limit: don't run more than once per _MIN_INTERVAL_SECONDS."""
        last = self._state.get('last_rebalance_at')
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last.replace('Z', ''))
            elapsed = (_utcnow() - last_dt).total_seconds()
            return elapsed >= _MIN_INTERVAL_SECONDS
        except Exception:
            return True

    def _should_force_rebalance(self):
        """True if >_MAX_INTERVAL_SECONDS since last rebalance."""
        last = self._state.get('last_rebalance_at')
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last.replace('Z', ''))
            elapsed = (_utcnow() - last_dt).total_seconds()
            return elapsed >= _MAX_INTERVAL_SECONDS
        except Exception:
            return True

    def maybe_rebalance(self):
        """Called by scheduler thread. Decides whether to actually rebalance
        based on: enabled, rotation, rate limit, drift, and time-since-last."""
        if not self.is_enabled():
            return {'skipped': True, 'reason': 'disabled'}

        self._refresh_targets()
        pending = False
        if self.auto_enabled():
            try:
                self._update_auto_targets()
            except Exception as e:
                log.warning(f"Auto targets update failed: {e}")
            pending = bool((self._state.get('auto') or {}).get('pending_rebalance'))
        self._sync_trading_pairs()

        # A rotation that just changed the targets bypasses the 6h rate limit.
        if not pending and not self._can_rebalance_now():
            return {'skipped': True, 'reason': 'rate limit (6h min between rebalances)'}

        status = self.get_status()
        force = self._should_force_rebalance()

        if not status['needs_rebalance'] and not force and not pending:
            return {'skipped': True, 'reason': f"drift {status['max_drift_pct']}% below threshold {self.drift_threshold_pct}%"}

        if pending:
            reason = 'auto-target rotation'
        elif force:
            reason = 'weekly forced rebalance'
        else:
            reason = f"drift {status['max_drift_pct']}% > threshold {self.drift_threshold_pct}%"
        return self.execute_rebalance(reason=reason)

    # ---------- Execute ----------

    def execute_rebalance(self, reason='manual'):
        """Do the rebalance NOW. Sells first (to free USDT), then buys.
        Individual trade failures are logged but don't abort the whole run.
        Returns a summary dict with all trades attempted."""
        log.info(f"Concentration REBALANCE starting - reason: {reason}")
        self._refresh_targets()
        self._sync_trading_pairs()
        status = self.get_status()
        total_value = status['total_portfolio_value']
        if total_value < 50:
            msg = f"Portfolio too small (${total_value}) for concentration rebalance - skipping"
            log.warning(msg)
            return {'skipped': True, 'reason': msg}

        targets = dict(self.targets)
        moves = []      # list of planned actions (for logging / telegram)
        trades = []     # list of executed trade results

        target_bases = {s.replace('USDT', '') for s in targets}
        current = self._compute_current_allocation(total_value)

        # Coins we're about to sell entirely (rotated out / never targets)
        sell_all_assets = [a for a, v in current.items()
                           if a not in target_bases and a not in _STABLECOINS and v >= _MIN_TRADE_USDT]

        # ---- Phase 0: cancel any existing OCOs on target pairs ----
        # Otherwise our sell orders would fail (asset is locked by OCO) and
        # the OCO's own take-profit could fire during the rebalance window.
        # Also covers coins about to be sold (e.g. rotated out).
        try:
            self._cancel_target_ocos(extra_symbols=[a + 'USDT' for a in sell_all_assets])
            time.sleep(1)  # brief pause for Binance to release locked balances
        except Exception as e:
            log.warning(f"Pre-rebalance OCO cancel failed (continuing): {e}")

        # ---- Phase 1: SELLS ----
        # (a) Non-target assets: sell 100%
        for asset in sell_all_assets:
            val = current[asset]
            pair = f"{asset}/USDT"
            moves.append({'action': 'sell_all', 'pair': pair, 'value_usdt': round(val, 2), 'reason': 'not in targets'})

        # (b) Over-target target pairs: sell down to target
        for sym, tgt_pct in targets.items():
            base = sym.replace('USDT', '')
            cur_val = current.get(base, 0.0)
            tgt_val = tgt_pct * total_value
            over_amount = cur_val - tgt_val
            if over_amount >= _MIN_TRADE_USDT:
                pair = f"{base}/USDT"
                moves.append({'action': 'sell_partial', 'pair': pair, 'value_usdt': round(over_amount, 2), 'reason': f'trim to {round(tgt_pct*100)}%'})

        # Send preview Telegram BEFORE executing so user sees plan even if trades fail
        try:
            self._tg_summary_preview(reason, moves, status)
        except Exception as e:
            log.debug(f"Preview telegram failed: {e}")

        # Execute sells
        for m in moves:
            try:
                pair = m['pair']
                if m['action'] == 'sell_all':
                    r = self.trader.execute_trade(pair, 'sell', 100.0, bypass_cooldown=True)
                else:
                    r = self.trader.execute_trade(pair, 'sell', 100.0,
                                                   amount_usdt=m['value_usdt'],
                                                   bypass_cooldown=True)
                trades.append({'pair': pair, 'action': m['action'], 'success': True, 'result': str(r)[:200]})
                log.info(f"Concentration SELL executed: {pair} ({m['action']}) ${m['value_usdt']}")
            except Exception as e:
                trades.append({'pair': m['pair'], 'action': m['action'], 'success': False, 'error': str(e)[:200]})
                log.warning(f"Concentration SELL failed for {m['pair']}: {e}")

        # Pause briefly so Binance balance settles before we start buying
        time.sleep(2)

        # ---- Phase 2: BUYS ----
        # Refetch to get post-sell allocation
        current_after = self._compute_current_allocation(total_value)
        # Note: total_value stays constant (rebalancing doesn't change total,
        # only distribution). Fine to reuse.

        buy_moves = []
        for sym, tgt_pct in targets.items():
            base = sym.replace('USDT', '')
            cur_val = current_after.get(base, 0.0)
            tgt_val = tgt_pct * total_value
            under_amount = tgt_val - cur_val
            if under_amount >= _MIN_TRADE_USDT:
                pair = f"{base}/USDT"
                buy_moves.append({'action': 'buy', 'pair': pair, 'value_usdt': round(under_amount, 2), 'reason': f'top up to {round(tgt_pct*100)}%'})

        for m in buy_moves:
            try:
                pair = m['pair']
                r = self.trader.execute_trade(pair, 'buy', 100.0,
                                              amount_usdt=m['value_usdt'],
                                              bypass_cooldown=True)
                trades.append({'pair': pair, 'action': 'buy', 'success': True, 'value_usdt': m['value_usdt']})
                log.info(f"Concentration BUY executed: {pair} ${m['value_usdt']}")
                moves.append(m)  # add to moves for final summary
            except Exception as e:
                trades.append({'pair': m['pair'], 'action': 'buy', 'success': False, 'error': str(e)[:200]})
                log.warning(f"Concentration BUY failed for {m['pair']}: {e}")

        # ---- Phase 3: cancel OCOs created by the buys ----
        # trader.execute_trade() auto-places OCO orders on every buy (using
        # regime TP/SL, currently 6%/4%). For concentration we want those
        # positions to HOLD - rebalance manages sizing via drift, not TP.
        # Wait briefly for OCOs to actually appear on Binance before cancel.
        try:
            time.sleep(2)
            n = self._cancel_target_ocos()
            if n:
                log.info(f"Concentration: removed {n} auto-OCOs so positions can hold")
        except Exception as e:
            log.warning(f"Post-rebalance OCO cancel failed: {e}")

        # Persist state. NOTE: update in place - replacing the whole dict
        # would wipe the performance baseline and the auto-target state.
        now_iso = _iso(_utcnow())
        history = self._state.get('history', [])
        history.append({
            'at': now_iso,
            'reason': reason,
            'total_value': total_value,
            'trades_count': len(trades),
            'trades_succeeded': sum(1 for t in trades if t.get('success')),
        })
        with self._lock:
            self._state['last_rebalance_at'] = now_iso
            self._state['history'] = history[-50:]  # keep last 50
            if isinstance(self._state.get('auto'), dict):
                self._state['auto']['pending_rebalance'] = False
            _save_state(self._state)

        # Post-execution telegram
        try:
            self._tg_summary_result(reason, trades)
        except Exception as e:
            log.debug(f"Result telegram failed: {e}")

        log.info(f"Concentration REBALANCE complete: {len(trades)} trades attempted, "
                 f"{sum(1 for t in trades if t.get('success'))} succeeded")
        return {
            'executed': True,
            'reason': reason,
            'trades': trades,
            'moves_planned': moves,
        }

    # ---------- Telegram helpers ----------

    def _tg_send(self, msg, context=None):
        """Route telegram via signal_engine if available (uses its dedup logic)."""
        if self.signal_engine and hasattr(self.signal_engine, '_tg_send'):
            try:
                self.signal_engine._tg_send(msg, context=context)
                return
            except Exception:
                pass
        # Fallback: try trader-level notification if available
        try:
            if hasattr(self.trader, 'send_telegram'):
                self.trader.send_telegram(msg)
        except Exception:
            pass

    def _tg_rotation(self, ev):
        out_s, in_s = _short(ev.get('out')), _short(ev.get('in'))
        w = ev.get('slot_weight_pct')
        kind = ev.get('type')
        os_ = ev.get('out_score')
        is_ = ev.get('in_score')
        if kind == 'swap':
            lines = [
                "🔄 *Concentration rotation*",
                f"{out_s} → {in_s} ({w}% slot)",
                f"Ranker score: {os_} → {is_} (+{round((is_ or 0) - (os_ or 0), 1)})",
                "Rebalance runs now.",
            ]
        elif kind == 'forced_exit':
            if ev.get('in'):
                lines = [
                    "🔄 *Concentration rotation*",
                    f"{out_s} fell below the score floor ({os_}) - replaced by {in_s} ({w}% slot)",
                    "Rebalance runs now.",
                ]
            else:
                lines = [
                    "⚠️ *Concentration: slot moved to cash*",
                    f"{out_s} fell below the score floor ({os_}) and nothing eligible can replace it.",
                    f"The {w}% slot stays in USDT until a coin qualifies.",
                ]
        else:  # fill
            lines = [
                "🔄 *Concentration rotation*",
                f"Empty {w}% slot filled with {in_s} (score {is_})",
                "Rebalance runs now.",
            ]
        self._tg_send("\n".join(lines), context=f"conc-rotation-{ev.get('at')}")

    def _tg_summary_preview(self, reason, moves, status):
        if not moves:
            return
        lines = [
            f"🎯 *Concentration Rebalance*",
            f"Reason: {reason}",
            f"Portfolio: ${status['total_portfolio_value']}",
            f"Max drift: {status['max_drift_pct']}%",
            "",
            "*Plan:*"
        ]
        for m in moves:
            emoji = '🔴' if m['action'].startswith('sell') else '🟢'
            lines.append(f"{emoji} {m['action']} {m['pair']} ${m['value_usdt']} ({m['reason']})")
        self._tg_send("\n".join(lines), context=f'conc-preview-{_utcnow().isoformat()}')

    def _tg_summary_result(self, reason, trades):
        if not trades:
            return
        succeeded = [t for t in trades if t.get('success')]
        failed = [t for t in trades if not t.get('success')]
        lines = [f"✅ *Rebalance done* ({len(succeeded)}/{len(trades)} trades)"]
        if failed:
            lines.append(f"⚠️ {len(failed)} failed:")
            for t in failed[:5]:
                lines.append(f"  - {t.get('pair')}: {t.get('error', '')[:80]}")
        self._tg_send("\n".join(lines), context=f'conc-result-{_utcnow().isoformat()}')

    # ---------- Background scheduler ----------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name='concentration_allocator')
        self._thread.start()
        log.info("Concentration allocator scheduler started (hourly checks)")

    def stop(self):
        self._stop = True

    def _run(self):
        # Wait 60s on startup so trader is fully initialized before first check
        time.sleep(60)
        # If concentration mode is already on when we boot (e.g. after a
        # redeploy), there may be leftover 6% OCOs from a previous rebalance.
        # Cancel them right away so positions can hold from now on.
        if self.is_enabled():
            try:
                self._refresh_targets()
                self._sync_trading_pairs()
                n = self._cancel_target_ocos()
                if n:
                    log.info(f"Concentration startup: removed {n} stale OCOs from previous rebalance")
            except Exception as e:
                log.debug(f"Startup OCO cleanup failed: {e}")
        while not self._stop:
            try:
                if self.is_enabled():
                    result = self.maybe_rebalance()
                    if result.get('executed'):
                        log.info(f"Concentration allocator ran a rebalance: {result.get('reason')}")
                    elif result.get('skipped'):
                        log.debug(f"Concentration check: skipped ({result.get('reason')})")
            except Exception as e:
                log.warning(f"Concentration scheduler tick failed: {e}")
            time.sleep(_CHECK_INTERVAL_SECONDS)

