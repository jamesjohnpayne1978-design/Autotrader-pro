"""
winners_allocator.py - Concentration allocator based on proven win rates.

Strategy (Phase 2 of the pivot from diversified swing-trading to concentrated
winners-focus). No momentum ranking, no signals - just target percentages
that get maintained.

Default targets (from user's actual 4-week trade history):
    SUI   40%  (88.9% historical win rate)
    BNB   25%  (83.3%)
    BTC   20%  (80.0%)
    USDT  15%  (dry powder)

Rebalance triggers (whichever hits first):
    - Any target pair drifts >5% (absolute) from its target percentage
    - OR 7 days elapsed since the last rebalance
    - AND at least 6 hours since the previous rebalance (rate limit)

Runs as a background thread alongside the existing signal engine. When
concentration_mode_enabled is True, the signal engine's auto-execute is
suppressed so the two don't fight each other. Manual trades still work.

Rebalance mechanics:
    1. Compute portfolio value + current allocation (from Binance balances)
    2. Compute deltas: target_value - current_value per asset
    3. Cancel open OCOs on all affected pairs (else sells will fail)
    4. Execute SELLS first: pairs not in targets get 100% sold, over-target
       target pairs get trimmed to target
    5. Wait a beat for USDT balance to settle
    6. Execute BUYS: under-target target pairs get topped up
    7. Send a single Telegram summary with all moves

Safety:
    - Skips if disabled
    - Skips if no drift over threshold and <7d elapsed
    - Rate-limited to 1 rebalance per 6h
    - Every trade goes through trader.execute_trade with bypass_cooldown=True
    - Individual trade failures do NOT abort the whole rebalance; they log
      and move on so partial rebalances still make progress
    - Telegram sent BEFORE execution (so user is warned even if a trade fails)
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

_STATE_PATH = '/data/concentration_state.json'
_MIN_TRADE_USDT = 10.0            # Skip rebalance moves below this ($)
_MIN_INTERVAL_SECONDS = 6 * 3600   # Rate limit: 6h between rebalances
_MAX_INTERVAL_SECONDS = 7 * 86400  # Force rebalance if 7 days elapsed
_CHECK_INTERVAL_SECONDS = 3600     # How often the scheduler wakes to check
_STABLECOINS = {'USDT', 'BUSD', 'USDC', 'FDUSD', 'TUSD', 'DAI'}


# Target allocations - fraction of total portfolio value.
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
        self._state = _load_state()

    # ---------- Enable / disable ----------

    def is_enabled(self):
        """Read the toggle from config. Kept as a method so it's re-checked
        every cycle - user can toggle mid-run and the next cycle picks it up."""
        return bool(getattr(self.config, 'concentration_mode_enabled', False))

    # ---------- State inspection (for API) ----------

    def get_status(self):
        """Return a snapshot for /api/concentration/status - dashboard reads
        this to show current vs target allocation, drift, and history."""
        enabled = self.is_enabled()
        try:
            portfolio = self.trader.get_portfolio()
            total_value = float(portfolio.get('total_usdt', 0) or 0)
        except Exception as e:
            log.debug(f"Concentration get_portfolio failed: {e}")
            total_value = 0.0

        current = self._compute_current_allocation(total_value)
        rows = []
        max_drift = 0.0
        for sym, tgt_pct in self.targets.items():
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
        cash_target_pct = (1.0 - sum(self.targets.values())) * 100
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
        target_bases = {s.replace('USDT', '') for s in self.targets}
        for asset, val in current.items():
            if asset in target_bases or asset in _STABLECOINS:
                continue
            if val >= _MIN_TRADE_USDT:
                non_target.append({'asset': asset, 'value_usdt': round(val, 2)})

        last_rebalance = self._state.get('last_rebalance_at')
        history = self._state.get('history', [])[-10:]  # last 10 events

        return {
            'enabled': enabled,
            'total_portfolio_value': round(total_value, 2),
            'drift_threshold_pct': self.drift_threshold_pct,
            'max_drift_pct': round(max_drift, 1),
            'needs_rebalance': max_drift > self.drift_threshold_pct,
            'targets': rows,
            'non_target_holdings': non_target,
            'last_rebalance_at': last_rebalance,
            'next_scheduled_check_hours': self._hours_until_next_forced_rebalance(),
            'history': history,
        }

    def _hours_until_next_forced_rebalance(self):
        """When will the 7-day timer force a rebalance regardless of drift?"""
        last = self._state.get('last_rebalance_at')
        if not last:
            return 0
        try:
            last_dt = datetime.fromisoformat(last.replace('Z', ''))
            elapsed = (datetime.utcnow() - last_dt).total_seconds()
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

    def _cancel_target_ocos(self):
        """Cancel any open orders on target pairs. Concentration mode holds
        positions long-term - it exits/sizes via rebalance drift, not via
        TP/SL. Any OCO on target coins would fire at 6% and force a costly
        re-entry loop (sell → drift alert → rebuy at market). Removing them
        lets positions ride freely; rebalance handles all sizing.

        Called BEFORE rebalance (to free locked balances for selling) and
        AFTER (to remove the new OCOs that trader.execute_trade auto-places
        on every buy).
        """
        cancelled = 0
        for sym in self.targets:
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
            log.info(f"Concentration: cancelled {cancelled} open OCO orders on target pairs")
        return cancelled

    # ---------- Rebalance decision ----------

    def _can_rebalance_now(self):
        """Rate limit: don't run more than once per _MIN_INTERVAL_SECONDS."""
        last = self._state.get('last_rebalance_at')
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last.replace('Z', ''))
            elapsed = (datetime.utcnow() - last_dt).total_seconds()
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
            elapsed = (datetime.utcnow() - last_dt).total_seconds()
            return elapsed >= _MAX_INTERVAL_SECONDS
        except Exception:
            return True

    def maybe_rebalance(self):
        """Called by scheduler thread. Decides whether to actually rebalance
        based on: enabled, rate limit, drift, and time-since-last."""
        if not self.is_enabled():
            return {'skipped': True, 'reason': 'disabled'}
        if not self._can_rebalance_now():
            return {'skipped': True, 'reason': 'rate limit (6h min between rebalances)'}

        status = self.get_status()
        force = self._should_force_rebalance()

        if not status['needs_rebalance'] and not force:
            return {'skipped': True, 'reason': f"drift {status['max_drift_pct']}% below threshold {self.drift_threshold_pct}%"}

        reason = 'weekly forced rebalance' if force else f"drift {status['max_drift_pct']}% > threshold {self.drift_threshold_pct}%"
        return self.execute_rebalance(reason=reason)

    # ---------- Execute ----------

    def execute_rebalance(self, reason='manual'):
        """Do the rebalance NOW. Sells first (to free USDT), then buys.
        Individual trade failures are logged but don't abort the whole run.
        Returns a summary dict with all trades attempted."""
        log.info(f"Concentration REBALANCE starting - reason: {reason}")
        status = self.get_status()
        total_value = status['total_portfolio_value']
        if total_value < 50:
            msg = f"Portfolio too small (${total_value}) for concentration rebalance - skipping"
            log.warning(msg)
            return {'skipped': True, 'reason': msg}

        moves = []      # list of planned actions (for logging / telegram)
        trades = []     # list of executed trade results

        target_bases = {s.replace('USDT', '') for s in self.targets}
        current = self._compute_current_allocation(total_value)

        # ---- Phase 0: cancel any existing OCOs on target pairs ----
        # Otherwise our sell orders would fail (asset is locked by OCO) and
        # the OCO's own take-profit could fire during the rebalance window.
        try:
            self._cancel_target_ocos()
            time.sleep(1)  # brief pause for Binance to release locked balances
        except Exception as e:
            log.warning(f"Pre-rebalance OCO cancel failed (continuing): {e}")

        # ---- Phase 1: SELLS ----
        # (a) Non-target assets: sell 100%
        for asset, val in list(current.items()):
            if asset in target_bases or asset in _STABLECOINS:
                continue
            if val < _MIN_TRADE_USDT:
                continue
            pair = f"{asset}/USDT"
            moves.append({'action': 'sell_all', 'pair': pair, 'value_usdt': round(val, 2), 'reason': 'not in targets'})

        # (b) Over-target target pairs: sell down to target
        for sym, tgt_pct in self.targets.items():
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
        for sym, tgt_pct in self.targets.items():
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

        # Persist state
        now_iso = datetime.utcnow().isoformat() + 'Z'
        history = self._state.get('history', [])
        history.append({
            'at': now_iso,
            'reason': reason,
            'total_value': total_value,
            'trades_count': len(trades),
            'trades_succeeded': sum(1 for t in trades if t.get('success')),
        })
        self._state = {
            'last_rebalance_at': now_iso,
            'history': history[-50:],  # keep last 50
        }
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
        self._tg_send("\n".join(lines), context=f'conc-preview-{datetime.utcnow().isoformat()}')

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
        self._tg_send("\n".join(lines), context=f'conc-result-{datetime.utcnow().isoformat()}')

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

