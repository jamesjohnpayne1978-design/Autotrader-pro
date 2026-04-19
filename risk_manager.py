"""
AutoTrader Pro - Risk Manager
Fixed: open positions count (ignores dust), max positions increased to 6
"""

import logging
from datetime import datetime

log = logging.getLogger(__name__)


class RiskManager:
    _trade_times = {}   # Class-level cooldown tracking persists across calls
    _trading_now = set()  # Pairs currently being traded (prevents simultaneous orders)
    def __init__(self, config):
        self.config = config

    def reload(self, config):
        self.config = config

    def check_trade(self, pair, action, confidence):
        """
        Returns (approved: bool, reason: str)
        """
        # Only check risk on buys
        if action == 'sell':
            return True, "Sell approved"

        # Prevent simultaneous orders for same pair
        if pair in RiskManager._trading_now:
            reason = f"Trade already in progress for {pair}"
            log.info(f"Risk check failed: {reason}")
            return False, reason
        RiskManager._trading_now.add(pair)

        # Check cooldown applies to buys too — prevent re-buying right after a sell
        if self._is_on_cooldown(pair):
            reason = f"Cooldown active for {pair} — preventing immediate re-buy"
            log.info(f"Risk check failed: {reason}")
            return False, reason

        # Check confidence threshold
        if confidence < 65:
            reason = f"Confidence {confidence}% below minimum 65%"
            log.info(f"Risk check failed for {pair}: {reason}")
            return False, reason

        # Check daily loss limit
        if self._daily_loss_exceeded():
            reason = "Daily loss limit reached — trading paused"
            log.warning(f"Risk check failed for {pair}: {reason}")
            return False, reason

        # Check if already holding this specific pair
        if self._already_holding(pair):
            reason = f"Already holding {pair} — will not buy more"
            log.info(f"Risk check failed for {pair}: {reason}")
            return False, reason

        # Check open positions — only count positions worth $1 or more
        open_count = self._count_open_positions()
        max_positions = getattr(self.config, 'max_open_positions', 6)
        if open_count >= max_positions:
            reason = f"Max open positions ({max_positions}) reached — currently {open_count} open"
            log.info(f"Risk check failed for {pair}: {reason}")
            return False, reason

        log.info(f"Risk check passed: {action} {pair} (confidence: {confidence}%)")
        return True, "Approved"

    def release_lock(self, pair):
        """Release trading lock after trade completes or fails"""
        RiskManager._trading_now.discard(pair)

    def _is_on_cooldown(self, pair):
        """Check cooldown using both memory AND saved trade history (survives restarts)"""
        from datetime import datetime
        cooldown = getattr(self.config, 'trade_cooldown_minutes', 60)

        # Check in-memory first (fastest)
        last = RiskManager._trade_times.get(pair)
        if last:
            elapsed = (datetime.now() - last).total_seconds() / 60
            if elapsed < cooldown:
                log.info(f"Cooldown (memory): {pair} traded {elapsed:.1f} mins ago")
                return True

        # Check saved trade history (survives restarts)
        try:
            history = self.config.load_trade_history()
            pair_name = pair.replace('USDT', '/USDT')
            recent = [t for t in history if t.get('pair') == pair_name]
            if recent:
                last_trade = recent[0]
                last_time_str = last_trade.get('time', '')
                if last_time_str:
                    # Try multiple datetime formats
                    last_dt = None
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S']:
                        try:
                            last_dt = datetime.strptime(last_time_str[:19], fmt)
                            break
                        except Exception:
                            continue
                    if last_dt:
                        elapsed = (datetime.now() - last_dt).total_seconds() / 60
                        if elapsed < cooldown:
                            log.info(f"Cooldown (history): {pair} traded {elapsed:.1f} mins ago, need {cooldown} mins")
                            RiskManager._trade_times[pair] = last_dt  # cache it
                            return True
        except Exception as e:
            log.debug(f"History cooldown check error: {e}")
        return False

    def record_trade(self, pair):
        """Call this after any trade to start cooldown"""
        from datetime import datetime
        RiskManager._trade_times[pair] = datetime.now()

    def _already_holding(self, pair):
        """Returns True if we already have a meaningful position in this pair"""
        try:
            from binance.client import Client
            client = Client(self.config.api_key, self.config.api_secret)
            base = pair.replace('/USDT', '').replace('USDT', '')
            account = client.get_account()
            for b in account['balances']:
                if b['asset'] == base:
                    amount = float(b['free']) + float(b['locked'])
                    if amount <= 0:
                        return False
                    ticker = client.get_symbol_ticker(symbol=f"{base}USDT")
                    value = amount * float(ticker['price'])
                    if value >= 1.0:
                        log.info(f"Already holding {base} worth ${value:.2f} — skipping buy")
                        return True
            return False
        except Exception as e:
            log.debug(f"Holdings check error: {e}")
            return False

    def _count_open_positions(self):
        """Count positions worth $1 or more — ignores dust amounts"""
        try:
            from binance.client import Client
            client = Client(self.config.api_key, self.config.api_secret)
            account = client.get_account()
            count = 0
            for b in account['balances']:
                asset = b['asset']
                if asset == 'USDT':
                    continue
                amount = float(b['free']) + float(b['locked'])
                if amount <= 0:
                    continue
                try:
                    ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
                    value = amount * float(ticker['price'])
                    if value >= 1.0:
                        count += 1
                except Exception:
                    pass
            return count
        except Exception as e:
            log.warning(f"Could not count positions: {e}")
            return 0

    def _daily_loss_exceeded(self):
        try:
            history = self.config.load_trade_history()
            today = datetime.now().strftime('%Y-%m-%d')
            today_trades = [t for t in history if t.get('date') == today]
            daily_pnl = sum(t.get('pnl', 0) for t in today_trades)
            limit = self.config.daily_loss_limit_pct

            # Get approximate portfolio value from history
            snapshots = self.config.load_portfolio_history()
            if snapshots:
                portfolio_value = snapshots[-1].get('value', 1000)
            else:
                portfolio_value = 1000

            loss_threshold = -(portfolio_value * limit / 100)
            if daily_pnl < loss_threshold:
                log.warning(f"Daily loss limit hit: {daily_pnl:.2f} < {loss_threshold:.2f}")
                return True
        except Exception as e:
            log.debug(f"Daily loss check error: {e}")
        return False
