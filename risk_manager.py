"""
AutoTrader Pro - Risk Manager
Protects your capital with configurable guardrails
"""

import logging
from datetime import datetime

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config):
        self.config = config

    def reload(self, config):
        self.config = config

    def check_trade(self, pair: str, action: str, confidence: float) -> tuple[bool, str]:
        """Check if a trade passes all risk rules"""

        # Minimum confidence threshold
        if confidence < 60:
            return False, f"Confidence {confidence}% below minimum 60%"

        # Check daily loss limit
        daily_pnl = self._get_daily_pnl()
        if daily_pnl < -abs(self.config.daily_loss_limit_pct):
            return False, f"Daily loss limit reached ({daily_pnl:.1f}%)"

        # Check if already in position
        if action == 'buy' and self._already_in_position(pair):
            return False, f"Already holding position in {pair}"

        # Check max concurrent positions
        open_positions = self._count_open_positions()
        if action == 'buy' and open_positions >= self.config.max_open_positions:
            return False, f"Max open positions ({self.config.max_open_positions}) reached"

        log.info(f"Risk check passed: {action} {pair} (confidence: {confidence}%)")
        return True, "OK"

    def check_snipe(self, symbol: str) -> tuple[bool, str]:
        """Check if a snipe trade is safe to execute"""

        # Check daily snipe budget
        daily_snipe_spend = self._get_daily_snipe_spend()
        if daily_snipe_spend + self.config.sniper_budget_usdt > self.config.sniper_daily_limit_usdt:
            return False, f"Daily snipe budget exhausted (${daily_snipe_spend:.0f} used)"

        # Check if already holding this coin
        base = symbol.replace('USDT', '')
        if self._already_in_position(f"{base}/USDT"):
            return False, f"Already holding {base}"

        return True, "OK"

    def _get_daily_pnl(self) -> float:
        """Get today's P&L as percentage"""
        try:
            history = self.config.load_trade_history()
            today = datetime.now().strftime('%Y-%m-%d')
            today_trades = [t for t in history if t.get('date') == today]
            total_pnl = sum(t.get('pnl', 0) for t in today_trades)
            return total_pnl  # Simplified — in production track as % of portfolio
        except Exception:
            return 0.0

    def _get_daily_snipe_spend(self) -> float:
        """Get today's total snipe spend in USDT"""
        try:
            history = self.config.load_trade_history()
            today = datetime.now().strftime('%Y-%m-%d')
            snipes = [t for t in history if t.get('date') == today and t.get('trigger') == 'Sniper']
            return sum(t.get('amount_usdt', 0) for t in snipes)
        except Exception:
            return 0.0

    def _already_in_position(self, pair: str) -> bool:
        """Check if we already have an open position"""
        try:
            history = self.config.load_trade_history()
            # Simplified: check if last trade for this pair was a buy with no matching sell
            pair_trades = [t for t in history if t.get('pair') == pair]
            if pair_trades and pair_trades[0].get('side') == 'buy':
                return True
        except Exception:
            pass
        return False

    def _count_open_positions(self) -> int:
        """Count current open positions"""
        try:
            history = self.config.load_trade_history()
            pairs = set(t['pair'] for t in history)
            open_count = 0
            for pair in pairs:
                if self._already_in_position(pair):
                    open_count += 1
            return open_count
        except Exception:
            return 0
