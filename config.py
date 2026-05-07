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
        # Mid cap — established coins (30 buy / 75 sell)
        self._rsi_mid_caps = {
            'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'LINKUSDT',
            'ADAUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT',
            'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'NEARUSDT',
            'APTUSDT', 'OPUSDT', 'INJUSDT', 'SUIUSDT',
            'TRXUSDT', 'XLMUSDT', 'VETUSDT', 'FILUSDT',
        }

        # Strategy toggles
        self.ma_cross_enabled = True
        self.macd_enabled = True

        # Trading mode
        self.auto_mode = os.environ.get('AUTO_MODE', 'false').lower() == 'true'
        self.approval_mode = True

        # Risk settings
        self.max_trade_pct = 5.0
        self.daily_loss_limit_pct = 5.0
        self.default_sl_pct = 4.0    # Wider — avoids normal volatility stopouts
        self.default_tp_pct = 12.0   # Bigger TP — let winners run
        self.dynamic_tp = 12.0        # Default dynamic TP
        # Trailing stop settings
        self.trailing_stop_enabled = True
        self.trailing_stop_pct = 2.5           # Trail 2.5% below highest price
        self.trailing_breakeven_trigger = 4.0  # Move to breakeven after 4% gain
        # Volume filter
        self.volume_filter_enabled = True
        self.volume_filter_min = 0.8           # Min 0.8x average volume to buy

        # Pyramiding settings
        self.pyramid_enabled = True
        self.pyramid_max_adds = 2          # Max 2 additional buys per pair (3 total)
        self.pyramid_drop_trigger = 4.0    # Add when price drops 4% from last buy
        self.pyramid_max_drop = 10.0       # Never add if down more than 10% from first buy
        self.pyramid_size_pct = 3.0        # Each add uses 3% of portfolio (smaller than initial 5%)
        self.max_open_positions = 6
        self.trade_cooldown_minutes = int(os.environ.get('TRADE_COOLDOWN_MINUTES', '60'))
        self.min_hold_minutes = 120  # Never auto-sell within 2 hours of buying

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

    def update(self, data):
        """Update config from settings dict"""
        if not data:
            return
        if 'max_trade_size' in data:
            self.max_trade_pct = float(data['max_trade_size'])
        if 'daily_loss_limit' in data:
            self.daily_loss_limit_pct = float(data['daily_loss_limit'])
        if 'default_sl' in data:
            self.default_sl_pct = float(data['default_sl'])
        if 'default_tp' in data:
            self.default_tp_pct = float(data['default_tp'])
        if 'auto_mode' in data:
            self.auto_mode = bool(data['auto_mode'])
        if 'ma_cross' in data:
            self.ma_cross_enabled = bool(data['ma_cross'])
        if 'macd' in data:
            self.macd_enabled = bool(data['macd'])
        if 'sniper_budget' in data:
            self.sniper_budget_usdt = float(data['sniper_budget'])
        if 'sniper_tp' in data:
            self.sniper_tp_pct = float(data['sniper_tp'])
        if 'sniper_sl' in data:
            self.sniper_sl_pct = float(data['sniper_sl'])
        if 'trade_cooldown_minutes' in data:
            self.trade_cooldown_minutes = int(data['trade_cooldown_minutes'])
        if 'pyramid_enabled' in data:
            self.pyramid_enabled = bool(data['pyramid_enabled'])
        if 'pyramid_max_adds' in data:
            self.pyramid_max_adds = int(data['pyramid_max_adds'])
        if 'pyramid_drop_trigger' in data:
            self.pyramid_drop_trigger = float(data['pyramid_drop_trigger'])
        if 'pyramid_max_drop' in data:
            self.pyramid_max_drop = float(data['pyramid_max_drop'])
        self.save()

    def save(self):
        """Save current settings to disk"""
        self.save_settings(self.to_dict())

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
            'trading_pairs': self.trading_pairs,
            'pyramid_enabled': self.pyramid_enabled,
            'pyramid_max_adds': self.pyramid_max_adds,
            'pyramid_drop_trigger': self.pyramid_drop_trigger,
            'pyramid_max_drop': self.pyramid_max_drop,
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
