"""
AutoTrader Pro - Configuration Manager
Added: portfolio history, trade cooldown, telegram config
"""

import json
import os
import logging

log = logging.getLogger(__name__)

CONFIG_PATH = '/data/config.json'
HISTORY_PATH = '/data/trade_history.json'
PORTFOLIO_HISTORY_PATH = '/data/portfolio_history.json'


class Config:
    def __init__(self):
        self.api_key = os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = os.environ.get('BINANCE_API_SECRET', '')

        log.info(f"API Key loaded: {'YES' if self.api_key else 'NO'}")
        log.info(f"API Secret loaded: {'YES' if self.api_secret else 'NO'}")

        # Read trading pairs from environment variable if set, otherwise use defaults
        pairs_env = os.environ.get('TRADING_PAIRS', '')
        if pairs_env:
            self.trading_pairs = [p.strip() for p in pairs_env.split(',') if p.strip()]
        else:
            self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'RENDERUSDT', 'SOLUSDT', 'LINKUSDT', 'ARBUSDT']
        self.rsi_buy = 35
        self.rsi_sell = 70
        # RSI tiers by market cap category — auto-assigns to any pair
        self._rsi_large_caps  = {'BTCUSDT', 'ETHUSDT'}
        self._rsi_mid_caps    = {'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT'}
        # All other pairs default to small cap thresholds

    def get_pair_rsi(self, symbol):
        """Auto-assign RSI thresholds based on pair tier. Works for any pair."""
        if symbol in self._rsi_large_caps:
            return (25, 80)   # Large cap — patient entries, hold longer
        elif symbol in self._rsi_mid_caps:
            return (30, 75)   # Mid cap — moderate thresholds
        else:
            return (32, 80)   # Small/speculative — tighter buy, hold for bigger move

    def _dummy(self):
        pass  # spacer
        self.ma_cross_enabled = True
        self.macd_enabled = True
        self.auto_mode = os.environ.get('AUTO_MODE', 'false').lower() == 'true'
        self.approval_mode = True
        self.max_trade_pct = 5.0
        self.daily_loss_limit_pct = 5.0
        self.default_sl_pct = 3.0
        self.default_tp_pct = 6.0
        self.max_open_positions = 6
        self.trade_cooldown_minutes = int(os.environ.get('TRADE_COOLDOWN_MINUTES', '60'))
        self.sniper_active = True
        self.sniper_budget_usdt = 50.0
        self.sniper_tp_pct = 20.0
        self.sniper_sl_pct = 10.0
        self.sniper_daily_limit_usdt = 200.0
        self.ai_filter_enabled = True
        self.ai_min_score = 70
        self.telegram_token = os.environ.get('TELEGRAM_TOKEN', '')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')

        self._load()

    def _load(self):
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    data = json.load(f)
                safe = {k: v for k, v in data.items()
                        if k not in ('api_key', 'api_secret')}
                self.update(safe)
        except Exception as e:
            log.warning(f"Could not load config: {e}")

    def update(self, data: dict):
        field_map = {
            'api_key': 'api_key',
            'api_secret': 'api_secret',
            'trading_pairs': 'trading_pairs',
            'rsi_buy': ('rsi_buy', float),
            'rsi_sell': ('rsi_sell', float),
            'ma_cross': ('ma_cross_enabled', bool),
            'macd': ('macd_enabled', bool),
            'auto_mode': ('auto_mode', bool),
            'approval_mode': ('approval_mode', bool),
            'max_trade_size': ('max_trade_pct', float),
            'daily_loss_limit': ('daily_loss_limit_pct', float),
            'default_sl': ('default_sl_pct', float),
            'default_tp': ('default_tp_pct', float),
            'sniper_active': ('sniper_active', bool),
            'sniper_budget': ('sniper_budget_usdt', float),
            'sniper_tp': ('sniper_tp_pct', float),
            'sniper_sl': ('sniper_sl_pct', float),
            'ai_filter': ('ai_filter_enabled', bool),
            'ai_min_score': ('ai_min_score', int),
            'trade_cooldown_minutes': ('trade_cooldown_minutes', int),
            'telegram_token': 'telegram_token',
            'telegram_chat_id': 'telegram_chat_id',
        }
        for key, val in data.items():
            if key in field_map:
                mapping = field_map[key]
                if isinstance(mapping, tuple):
                    attr, cast = mapping
                    try:
                        setattr(self, attr, cast(val))
                    except Exception:
                        pass
                else:
                    setattr(self, mapping, val)

    def save(self):
        try:
            os.makedirs('/data', exist_ok=True)
            with open(CONFIG_PATH, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            log.warning(f"Config save failed: {e}")

    def to_dict(self):
        return {
            'trading_pairs': self.trading_pairs,
            'rsi_buy': self.rsi_buy,
            'rsi_sell': self.rsi_sell,
            'ma_cross': self.ma_cross_enabled,
            'macd': self.macd_enabled,
            'auto_mode': self.auto_mode,
            'approval_mode': self.approval_mode,
            'max_trade_size': self.max_trade_pct,
            'daily_loss_limit': self.daily_loss_limit_pct,
            'default_sl': self.default_sl_pct,
            'default_tp': self.default_tp_pct,
            'sniper_active': self.sniper_active,
            'sniper_budget': self.sniper_budget_usdt,
            'sniper_tp': self.sniper_tp_pct,
            'sniper_sl': self.sniper_sl_pct,
            'ai_filter': self.ai_filter_enabled,
            'ai_min_score': self.ai_min_score,
            'trade_cooldown_minutes': self.trade_cooldown_minutes,
        }

    def load_trade_history(self):
        try:
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def save_trade_history(self, history):
        try:
            os.makedirs('/data', exist_ok=True)
            with open(HISTORY_PATH, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            log.error(f"History save failed: {e}")

    def load_portfolio_history(self):
        try:
            if os.path.exists(PORTFOLIO_HISTORY_PATH):
                with open(PORTFOLIO_HISTORY_PATH, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def save_portfolio_history(self, history):
        try:
            os.makedirs('/data', exist_ok=True)
            with open(PORTFOLIO_HISTORY_PATH, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            log.error(f"Portfolio history save failed: {e}")
