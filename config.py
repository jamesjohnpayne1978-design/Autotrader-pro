import os
import json
import logging

log = logging.getLogger(__name__)

DATA_DIR = os.environ.get('DATA_DIR', '/data')
HISTORY_FILE = os.path.join(DATA_DIR, 'trade_history.json')
PORTFOLIO_FILE = os.path.join(DATA_DIR, 'portfolio_history.json')
SETTINGS_FILE = os.path.join(DATA_DIR, 'settings.json')

class Config:
    def __init__(self):
        self.api_key = os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = os.environ.get('BINANCE_API_SECRET', '')

        log.info(f"API Key loaded: {'YES' if self.api_key else 'NO'}")
        log.info(f"API Secret loaded: {'YES' if self.api_secret else 'NO'}")

        # Trading pairs
        pairs_env = os.environ.get('TRADING_PAIRS', '')
        if pairs_env:
            self.trading_pairs = [p.strip() for p in pairs_env.split(',') if p.strip()]
        else:
            self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'RENDERUSDT', 'SOLUSDT', 'LINKUSDT', 'ARBUSDT']

        # RSI defaults (used as fallback only — tier system overrides these)
        self.rsi_buy = 35
        self.rsi_sell = 70

        # RSI tier sets — auto-assigns thresholds by market cap
        self._rsi_large_caps = {'BTCUSDT', 'ETHUSDT'}
        self._rsi_mid_caps = {'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT'}

        # Strategy toggles
        self.ma_cross_enabled = True
        self.macd_enabled = True

        # Trading mode
        self.auto_mode = os.environ.get('AUTO_MODE', 'false').lower() == 'true'
        self.approval_mode = True

        # Risk settings
        self.max_trade_pct = 5.0
        self.daily_loss_limit_pct = 5.0
        self.default_sl_pct = 3.0
        self.default_tp_pct = 6.0
        self.dynamic_tp = 6.0
        self.max_open_positions = 6
        self.trade_cooldown_minutes = int(os.environ.get('TRADE_COOLDOWN_MINUTES', '60'))

        # Sniper settings
        self.sniper_active = True
        self.sniper_budget_usdt = 50.0
        self.sniper_tp_pct = 20.0
        self.sniper_sl_pct = 10.0
        self.sniper_daily_limit_usdt = 200.0
        self.ai_filter_enabled = True
        self.ai_min_score = 70

        # Telegram
        self.telegram_token = os.environ.get('TELEGRAM_TOKEN', '')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')

        # Load saved settings over defaults
        self._load_saved_settings()

    def get_pair_rsi(self, symbol):
        """Auto-assign RSI thresholds based on pair tier. Works for any new pair."""
        if symbol in self._rsi_large_caps:
            return (25, 80)   # Large cap — patient entries
        elif symbol in self._rsi_mid_caps:
            return (30, 75)   # Mid cap — moderate thresholds
        else:
            return (32, 80)   # Small/speculative — tighter entries

    def _load_saved_settings(self):
        """Load persisted settings from disk"""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE) as f:
                    s = json.load(f)
                if 'max_trade_size' in s:
                    self.max_trade_pct = float(s['max_trade_size'])
                if 'daily_loss_limit' in s:
                    self.daily_loss_limit_pct = float(s['daily_loss_limit'])
                if 'default_sl' in s:
                    self.default_sl_pct = float(s['default_sl'])
                if 'default_tp' in s:
                    self.default_tp_pct = float(s['default_tp'])
                    self.dynamic_tp = float(s['default_tp'])
                if 'auto_mode' in s:
                    self.auto_mode = bool(s['auto_mode'])
                if 'ma_cross' in s:
                    self.ma_cross_enabled = bool(s['ma_cross'])
                if 'macd' in s:
                    self.macd_enabled = bool(s['macd'])
                if 'sniper_budget' in s:
                    self.sniper_budget_usdt = float(s['sniper_budget'])
                if 'sniper_tp' in s:
                    self.sniper_tp_pct = float(s['sniper_tp'])
                if 'sniper_sl' in s:
                    self.sniper_sl_pct = float(s['sniper_sl'])
        except Exception as e:
            log.warning(f"Could not load saved settings: {e}")

    def save_settings(self, settings_dict):
        """Persist settings to disk"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings_dict, f, indent=2)
            self._load_saved_settings()
            return True
        except Exception as e:
            log.error(f"Could not save settings: {e}")
            return False

    def to_dict(self):
        """Return current config as dict for API"""
        return {
            'max_trade_size': self.max_trade_pct,
            'daily_loss_limit': self.daily_loss_limit_pct,
            'default_sl': self.default_sl_pct,
            'default_tp': self.default_tp_pct,
            'auto_mode': self.auto_mode,
            'ma_cross': self.ma_cross_enabled,
            'macd': self.macd_enabled,
            'sniper_budget': self.sniper_budget_usdt,
            'sniper_tp': self.sniper_tp_pct,
            'sniper_sl': self.sniper_sl_pct,
            'trade_cooldown_minutes': self.trade_cooldown_minutes,
            'rsi_buy': self.rsi_buy,
            'rsi_sell': self.rsi_sell,
        }

    def load_trade_history(self):
        """Load saved trade history from disk"""
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE) as f:
                    return json.load(f)
        except Exception as e:
            log.warning(f"Could not load trade history: {e}")
        return []

    def save_trade_history(self, history):
        """Save trade history to disk"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            log.error(f"Could not save trade history: {e}")

    def load_portfolio_history(self):
        """Load portfolio snapshots from disk"""
        try:
            if os.path.exists(PORTFOLIO_FILE):
                with open(PORTFOLIO_FILE) as f:
                    return json.load(f)
        except Exception as e:
            log.warning(f"Could not load portfolio history: {e}")
        return []

    def save_portfolio_history(self, history):
        """Save portfolio snapshots to disk"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            log.error(f"Could not save portfolio history: {e}")
