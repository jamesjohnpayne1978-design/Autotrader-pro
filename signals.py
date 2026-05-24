"""
AutoTrader Pro - AI Signal Engine
Added: Market regime detection + dynamic take profit adjustment
AI provider chain: OpenAI (primary) -> Gemini (fallback) -> rule-based
"""

import os
import time
import logging
import json
import numpy as np
import requests
from datetime import datetime
from config import Config

log = logging.getLogger(__name__)

# ============================================================
# AI PROVIDER ENDPOINTS AND MODELS
# ============================================================
OPENAI_API   = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

GEMINI_API   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
# ============================================================


AUTO_EXECUTE_MIN_CONFIDENCE = 60
PYRAMID_MIN_CONFIDENCE      = 65
MIN_HOLD_MINUTES_BEFORE_AI_SELL = 240
_BUY_TIMES_PATH = '/data/buy_times.json'

_EXTRA_SETTINGS_PATH = '/data/extra_settings.json'
_extra_cache = {'data': {}, 'mtime': 0}


def _read_extras():
    try:
        if not os.path.exists(_EXTRA_SETTINGS_PATH):
            return {}
        mtime = os.path.getmtime(_EXTRA_SETTINGS_PATH)
        if mtime != _extra_cache['mtime']:
            with open(_EXTRA_SETTINGS_PATH) as f:
                _extra_cache['data'] = json.load(f) or {}
            _extra_cache['mtime'] = mtime
            log.info(f"Extras reloaded from disk: {list(_extra_cache['data'].keys())}")
        return _extra_cache['data']
    except Exception as e:
        log.debug(f"Could not read extras: {e}")
        return _extra_cache.get('data', {})


def _extra_bool(key, default=False):
    v = _read_extras().get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == 'true'
    return bool(v)


def _extra_float(key, default):
    v = _read_extras().get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def _openai_key():
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _gemini_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _call_ai(prompt, max_tokens=500, _err_state=None):
    okey = _openai_key()
    if okey:
        try:
            r = requests.post(
                OPENAI_API,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {okey}"},
                json={"model": OPENAI_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": 0.4},
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            else:
                _log_ai_error("OpenAI", r.status_code, r.text, _err_state)
        except Exception as e:
            _log_ai_error("OpenAI", "exc", str(e), _err_state)

    gkey = _gemini_key()
    if gkey:
        try:
            r = requests.post(
                f"{GEMINI_API}?key={gkey}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_tokens}},
                timeout=30,
            )
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                text = text.replace("```json", "").replace("```", "").strip()
                return text
            else:
                _log_ai_error("Gemini", r.status_code, r.text, _err_state)
        except Exception as e:
            _log_ai_error("Gemini", "exc", str(e), _err_state)
    return None


def _log_ai_error(provider, status, body, state):
    if state is None:
        return
    state["count"] = state.get("count", 0) + 1
    if state["count"] <= 3:
        msg = body[:200] if isinstance(body, str) else body
        log.warning(f"AI provider {provider} failed (status={status}): {msg}")
    elif state["count"] == 4:
        log.warning(f"AI provider {provider} continuing to fail - suppressing further error logs")


def _extract_json_block(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


class SignalEngine:
    def __init__(self, config, trader, risk_manager=None, manual_manager=None):
        self.config = config
        self.trader = trader
        self.risk_manager = risk_manager
        self.manual_manager = manual_manager
        self.latest_signals = []
        self.market_regime = 'neutral'
        self.fear_greed_index = 50
        self.fear_greed_label = 'Neutral'
        self.regime_reason = 'Analysing market conditions...'
        self.regime_tp = 6.0
        self.running = False
        self._ai_err_state = {"count": 0}
        self._last_buy_times = self._load_buy_times()

        openai = bool(_openai_key())
        gemini = bool(_gemini_key())
        if openai and gemini:
            log.info(f"AI providers configured: OpenAI/{OPENAI_MODEL} (primary) + Gemini (fallback)")
        elif openai:
            log.info(f"AI provider configured: OpenAI/{OPENAI_MODEL}")
        elif gemini:
            log.info("AI provider configured: Gemini (free tier)")
        else:
            log.warning("NO AI PROVIDER CONFIGURED - all signals will use rule-based fallback.")

    def _load_buy_times(self):
        try:
            with open(_BUY_TIMES_PATH, 'r') as f:
                data = json.load(f) or {}
                log.info(f"Buy times loaded from disk: {len(data)} pairs")
                return data
        except FileNotFoundError:
            log.info("No buy times file - seeding from Binance trade history")
            return self._seed_buy_times_from_binance()
        except Exception as e:
            log.warning(f"Could not load buy times: {e}")
            return {}

    def _seed_buy_times_from_binance(self):
        times = {}
        if not self.trader:
            return times
        try:
            for symbol in self.config.trading_pairs:
                pair = symbol.replace('USDT', '/USDT')
                try:
                    trades = self.trader.client.get_my_trades(symbol=symbol, limit=20)
                    buys = [t for t in trades if t.get('isBuyer')]
                    if buys:
                        last_buy = max(buys, key=lambda t: int(t['time']))
                        ts = datetime.fromtimestamp(int(last_buy['time']) / 1000)
                        times[pair] = ts.isoformat()
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Buy time seeding failed: {e}")
        if times:
            log.info(f"Seeded buy times from Binance for {len(times)} pairs")
            self._last_buy_times = times
            self._save_buy_times()
        return times

    def _save_buy_times(self):
        try:
            os.makedirs(os.path.dirname(_BUY_TIMES_PATH), exist_ok=True)
            with open(_BUY_TIMES_PATH, 'w') as f:
                json.dump(self._last_buy_times, f)
        except Exception as e:
            log.warning(f"Could not save buy times: {e}")

    def _record_buy_time(self, pair):
        self._last_buy_times[pair] = datetime.now().isoformat()
        self._save_buy_times()

    def _check_min_hold(self, pair):
        if MIN_HOLD_MINUTES_BEFORE_AI_SELL <= 0:
            return None
        last_buy = self._last_buy_times.get(pair)
        if not last_buy:
            return None
        try:
            last_buy_dt = datetime.fromisoformat(last_buy)
            elapsed_min = (datetime.now() - last_buy_dt).total_seconds() / 60
            if elapsed_min < MIN_HOLD_MINUTES_BEFORE_AI_SELL:
                remaining = MIN_HOLD_MINUTES_BEFORE_AI_SELL - elapsed_min
                return (f"hold period active ({elapsed_min:.0f} min since buy, "
                        f"need {MIN_HOLD_MINUTES_BEFORE_AI_SELL}) - "
                        f"{remaining:.0f} min remaining. TP/SL still active.")
        except Exception:
            pass
        return None

    def run(self):
        self.running = True
        log.info("Signal engine started.")
        while self.running:
            if not hasattr(self, '_fg_count'):
                self._fg_count = 0
            self._fg_count += 1
            if self._fg_count % 6 == 1:
                self._fetch_fear_greed()

            try:
                self.detect_market_regime()
                try:
                    self._check_portfolio_trailing_stops()
                except Exception as e:
                    log.debug(f"Portfolio trailing stop error: {e}")
                self.refresh_signals()
                try:
                    if self.trader:
                        self.trader.check_all_trailing_stops()
                except Exception as e:
                    log.debug(f"Trailing stop check error: {e}")
                try:
                    self._check_position_concentration()
                except Exception as e:
                    log.debug(f"Concentration check error: {e}")
            except Exception as e:
                log.error(f"Signal refresh error: {e}")
            time.sleep(300)

    def _check_portfolio_trailing_stops(self):
        if not _extra_bool('trailing_stop_enabled', False):
            return
        if not self.trader:
            return

        if not hasattr(self, '_trail_state_loaded'):
            self._trail_state_loaded = True
            self.high_water_marks = self._load_trailing_stops()

        if not hasattr(self, 'high_water_marks'):
            self.high_water_marks = {}

        trail_pct = _extra_float('trailing_stop_pct', 5.0)
        activate_pct = _extra_float('trailing_stop_activate_pct', 2.0)
        breakeven_trigger = _extra_float('trailing_breakeven_trigger', 3.0)

        try:
            account = self.trader.client.get_account()
            prices = self.trader.client.get_all_tickers()
            price_map = {p['symbol']: float(p['price']) for p in prices}
            state_changed = False

            held_symbols = set()
            for balance in account['balances']:
                asset = balance['asset']
                if asset in ('USDT', 'BNB', 'BUSD', 'USDC', 'FDUSD'):
                    continue
                total = float(balance['free']) + float(balance['locked'])
                if total <= 0:
                    continue
                symbol = f"{asset}USDT"
                if symbol not in price_map:
                    continue
                current_price = price_map[symbol]
                value = total * current_price
                if value < 5.0:
                    continue
                held_symbols.add(symbol)

                if symbol not in self.high_water_marks:
                    entry_price = self._get_entry_price_from_binance(symbol) or current_price
                    self.high_water_marks[symbol] = {
                        'entry_price': entry_price,
                        'peak_price': max(entry_price, current_price),
                        'stop_price': entry_price * (1 - float(getattr(self.config, 'default_sl', getattr(self.config, 'default_sl_pct', 5.0))) / 100),
                    }
                    state_changed = True
                    log.info(f"Trailing stop init {symbol}: entry ${entry_price:.6f}, current ${current_price:.6f}")
                    continue

                hwm = self.high_water_marks[symbol]
                entry_price = hwm['entry_price']

                if current_price > hwm['peak_price']:
                    hwm['peak_price'] = current_price
                    state_changed = True

                gain_from_entry = ((current_price - entry_price) / entry_price) * 100
                drop_from_peak = ((hwm['peak_price'] - current_price) / hwm['peak_price']) * 100

                if gain_from_entry < activate_pct:
                    continue

                if gain_from_entry >= breakeven_trigger:
                    new_stop = entry_price * 1.001
                    if new_stop > hwm.get('stop_price', 0):
                        hwm['stop_price'] = new_stop
                        state_changed = True
                        log.info(f"Trailing stop {symbol}: moved to breakeven @ ${new_stop:.6f}")

                trailed_stop = hwm['peak_price'] * (1 - trail_pct / 100)
                if trailed_stop > hwm.get('stop_price', 0):
                    hwm['stop_price'] = trailed_stop
                    state_changed = True
                    log.info(f"Trailing stop {symbol}: trailed to ${trailed_stop:.6f}")

                pair_slash = symbol.replace('USDT', '/USDT')
                try:
                    if self.manual_manager and self.manual_manager.has_position(pair_slash):
                        continue
                except Exception:
                    pass

                if current_price <= hwm.get('stop_price', 0):
                    locked_in = ((hwm['stop_price'] - entry_price) / entry_price) * 100
                    log.info(f"TRAILING STOP FIRED for {symbol}: locked-in {locked_in:+.1f}%")
                    try:
                        if self.risk_manager:
                            approved, _ = self.risk_manager.check_trade(pair_slash, 'sell', 100)
                            if not approved:
                                continue
                        try:
                            self.trader._cancel_all_open_orders(symbol)
                        except Exception:
                            pass
                        result = self.trader.execute_trade(pair_slash, 'sell', 100.0)
                        log.info(f"Trailing stop SELL executed for {pair_slash}: {result}")
                        del self.high_water_marks[symbol]
                        state_changed = True
                        if self.risk_manager:
                            self.risk_manager.release_lock(pair_slash)
                    except Exception as e:
                        log.error(f"Trailing stop sell failed for {pair_slash}: {e}")
                        if self.risk_manager:
                            try:
                                self.risk_manager.release_lock(pair_slash)
                            except Exception:
                                pass

            for symbol in list(self.high_water_marks.keys()):
                if symbol not in held_symbols:
                    log.info(f"Trailing stop cleared {symbol}")
                    del self.high_water_marks[symbol]
                    state_changed = True

            if state_changed:
                self._save_trailing_stops()
        except Exception as e:
            log.debug(f"Portfolio trailing stop scan error: {e}")

    def _get_entry_price_from_binance(self, symbol):
        try:
            trades = self.trader.client.get_my_trades(symbol=symbol, limit=50)
            trades = sorted(trades, key=lambda t: int(t['time']))
            last_sell_time = None
            for t in reversed(trades):
                if not t['isBuyer']:
                    last_sell_time = int(t['time'])
                    break
            unmatched = []
            for t in trades:
                if t['isBuyer'] and (last_sell_time is None or int(t['time']) > last_sell_time):
                    unmatched.append({'price': float(t['price']), 'qty': float(t['qty'])})
            if not unmatched:
                return None
            total_qty = sum(b['qty'] for b in unmatched)
            if total_qty <= 0:
                return None
            return sum(b['price'] * b['qty'] for b in unmatched) / total_qty
        except Exception as e:
            log.debug(f"Could not fetch entry price for {symbol}: {e}")
            return None

    _TRAILING_STOPS_PATH = '/data/trailing_stops.json'

    def _load_trailing_stops(self):
        try:
            if os.path.exists(self._TRAILING_STOPS_PATH):
                with open(self._TRAILING_STOPS_PATH, 'r') as f:
                    data = json.load(f) or {}
                    log.info(f"Trailing stops loaded: {len(data)} positions")
                    return data
        except Exception as e:
            log.warning(f"Could not load trailing stops: {e}")
        return {}

    def _save_trailing_stops(self):
        try:
            os.makedirs(os.path.dirname(self._TRAILING_STOPS_PATH), exist_ok=True)
            with open(self._TRAILING_STOPS_PATH, 'w') as f:
                json.dump(self.high_water_marks, f)
        except Exception as e:
            log.warning(f"Could not save trailing stops: {e}")

    def _check_position_concentration(self):
        if not self.trader:
            return
        threshold_pct = _extra_float('concentration_alert_pct', 25.0)
        cooldown_hours = 4
        if not hasattr(self, '_last_concentration_alert'):
            self._last_concentration_alert = {}
        try:
            account = self.trader.client.get_account()
            prices = self.trader.client.get_all_tickers()
            price_map = {p['symbol']: float(p['price']) for p in prices}
            holdings = {}
            usdt_balance = 0.0
            for balance in account['balances']:
                asset = balance['asset']
                total = float(balance['free']) + float(balance['locked'])
                if total <= 0:
                    continue
                if asset == 'USDT':
                    usdt_balance = total
                    continue
                symbol = f"{asset}USDT"
                if symbol not in price_map:
                    continue
                value = total * price_map[symbol]
                if value >= 1.0:
                    holdings[asset] = value
            total_value = sum(holdings.values()) + usdt_balance
            if total_value < 10:
                return
            now = time.time()
            for asset, value in holdings.items():
                pct = (value / total_value) * 100
                if pct < threshold_pct:
                    continue
                last_alert = self._last_concentration_alert.get(asset, 0)
                if now - last_alert < cooldown_hours * 3600:
                    continue
                log.info(f"Concentration alert: {asset} is {pct:.1f}%")
                self._last_concentration_alert[asset] = now
                try:
                    if self.config.telegram_token and self.config.telegram_chat_id:
                        msg = (f"⚖️ *CONCENTRATION ALERT*\n{asset} is *{pct:.1f}%* of portfolio.")
                        requests.post(
                            f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                            json={'chat_id': str(self.config.telegram_chat_id), 'text': msg, 'parse_mode': 'Markdown'},
                            timeout=8)
                except Exception:
                    pass
        except Exception as e:
            log.debug(f"Concentration check error: {e}")

    def _fetch_fear_greed(self):
        try:
            r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=8)
            if r.status_code == 200:
                data = r.json()
                value = int(data['data'][0]['value'])
                classification = data['data'][0]['value_classification']
                self.fear_greed_index = value
                self.fear_greed_label = classification
                log.info(f"Fear & Greed: {value} ({classification})")
                return value
        except Exception as e:
            log.debug(f"Fear & Greed fetch error: {e}")
        return getattr(self, 'fear_greed_index', 50)

    def detect_market_regime(self):
        try:
            klines = self.trader.get_klines('BTCUSDT', '1d', 14)
            if len(klines) < 7:
                return
            closes = [k['close'] for k in klines]
            volumes = [k['volume'] for k in klines]
            rsi = self._rsi(closes)
            ma7 = np.mean(closes[-7:])
            ma14 = np.mean(closes)
            price = closes[-1]
            price_7d_change = ((closes[-1] - closes[-7]) / closes[-7]) * 100
            volume_trend = np.mean(volumes[-3:]) / np.mean(volumes[-14:])

            prompt = (
                f"You are a crypto market analyst. Determine the current market regime "
                f"based on these Bitcoin indicators:\n\n"
                f"Current price: ${round(price, 0)}\n"
                f"7-day price change: {round(price_7d_change, 2)}%\n"
                f"RSI (14): {round(rsi, 1)}\n"
                f"Price vs 7-day MA: {'above' if price > ma7 else 'below'}\n"
                f"Price vs 14-day MA: {'above' if price > ma14 else 'below'}\n"
                f"Volume trend: {round(volume_trend, 2)}x\n"
                f"F&G: {self.fear_greed_index}/100 ({self.fear_greed_label})\n\n"
                f"Determine market regime and recommended TP.\n"
                f"Return ONLY valid JSON, no markdown:\n"
                f'{{"regime":"bullish|bearish|neutral","take_profit":6,"reason":"one sentence"}}'
            )

            text = _call_ai(prompt, max_tokens=200, _err_state=self._ai_err_state)
            result = _extract_json_block(text) if text else None

            if result:
                self.market_regime = result.get('regime', 'neutral')
                self.regime_tp = float(result.get('take_profit', 6.0))
                self.regime_reason = result.get('reason', '')
                self.config.dynamic_tp = self.regime_tp
                log.info(f"Market regime (AI): {self.market_regime} - TP {self.regime_tp}%")
                return

            self._fallback_regime(rsi, price_7d_change, price, ma7)
        except Exception as e:
            log.warning(f"Regime detection failed: {e}")
            self._fallback_regime_simple()

    def _fallback_regime(self, rsi, price_change_7d, price, ma7):
        if rsi > 55 and price_change_7d > 3 and price > ma7:
            self.market_regime = 'bullish'
            self.regime_tp = 15.0
            self.regime_reason = f"RSI {round(rsi)} above 55, price up {round(price_change_7d, 1)}%."
        elif rsi < 45 and price_change_7d < -3 and price < ma7:
            self.market_regime = 'bearish'
            self.regime_tp = 8.0
            self.regime_reason = f"RSI {round(rsi)} below 45, price down {round(abs(price_change_7d), 1)}%."
        else:
            self.market_regime = 'neutral'
            self.regime_tp = 6.0
            self.regime_reason = f"Mixed signals - RSI {round(rsi)}."
        self.config.dynamic_tp = self.regime_tp
        log.info(f"Market regime (rule-based): {self.market_regime} - TP {self.regime_tp}%")

    def _fallback_regime_simple(self):
        self.market_regime = 'neutral'
        self.fear_greed_index = 50
        self.fear_greed_label = 'Neutral'
        self.regime_tp = 6.0
        self.regime_reason = "Using default settings."
        self.config.dynamic_tp = self.regime_tp

    def get_regime(self):
        return {'regime': self.market_regime, 'take_profit': self.regime_tp, 'reason': self.regime_reason}

    def refresh_signals(self):
        if self.risk_manager:
            for symbol in self.config.trading_pairs:
                pair_slash = symbol.replace('USDT', '/USDT')
                try:
                    self.risk_manager.release_lock(pair_slash)
                except Exception:
                    pass

        signals = []
        for symbol in self.config.trading_pairs:
            try:
                signal = self.analyse_pair(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                log.warning(f"Signal failed for {symbol}: {e}")
        self.latest_signals = signals
        log.info(f"Signals refreshed: {len(signals)} signals - regime: {self.market_regime}")
        if self.config.auto_mode and self.risk_manager:
            self._auto_execute(signals)

    def _check_pyramid_opportunity(self, signal):
        pair = signal.get('pair')
        action = signal.get('action', 'hold')
        confidence = signal.get('confidence', 0)
        if action == 'sell':
            return
        if confidence < PYRAMID_MIN_CONFIDENCE:
            return
        try:
            if self.manual_manager and self.manual_manager.has_position(pair):
                return
        except Exception:
            pass
        try:
            sym = pair.replace('/', '')
            prices = self.trader.client.get_symbol_ticker(symbol=sym)
            current_price = float(prices['price'])
            should_add, reason = self.trader.should_pyramid(sym, current_price)
        except Exception as e:
            log.warning(f"Pyramid should_pyramid check failed for {pair}: {e}")
            return
        if not should_add:
            return
        approved, rm_reason = self.risk_manager.check_trade(pair, 'buy', confidence)
        if not approved:
            log.info(f"Pyramid {pair} blocked by risk manager: {rm_reason}")
            try:
                self.risk_manager.release_lock(pair)
            except Exception:
                pass
            return
        try:
            log.info(f"Pyramid opportunity for {pair}: {reason}")
            pyramid_pct = getattr(self.config, 'pyramid_size_pct', 3.0)
            result = self.trader.execute_trade(pair, 'buy', pyramid_pct)
            self.risk_manager.record_trade(pair)
            self._record_buy_time(pair)
            log.info(f"Pyramid buy executed for {pair}: {result}")
        except Exception as e:
            log.error(f"Pyramid execute failed for {pair}: {e}")
        finally:
            try:
                self.risk_manager.release_lock(pair)
            except Exception:
                pass

    def _notify_pending_signals(self, signals):
        if not hasattr(self, '_last_notified_signal'):
            self._last_notified_signal = {}
        pending = []
        for sig in signals:
            action = sig.get('action')
            conf = sig.get('confidence', 0)
            pair = sig.get('pair', '')
            if action not in ('buy', 'sell') or conf < AUTO_EXECUTE_MIN_CONFIDENCE:
                continue
            last = self._last_notified_signal.get(pair, {})
            if last.get('action') == action and abs(last.get('confidence', 0) - conf) < 10:
                continue
            self._last_notified_signal[pair] = {'action': action, 'confidence': conf}
            pending.append(sig)
        if not pending:
            return
        try:
            tok = getattr(self.config, 'telegram_token', '') or ''
            chat = getattr(self.config, 'telegram_chat_id', '') or ''
            if not tok or not chat:
                return
            lines = ["⏸️ *APPROVAL NEEDED*"]
            for sig in pending:
                icon = '🟢' if sig['action'] == 'buy' else '🔴'
                lines.append(f"{icon} *{sig['action'].upper()} {sig['pair']}* — {sig['confidence']}%")
            requests.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                json={'chat_id': str(chat), 'text': '\n'.join(lines), 'parse_mode': 'Markdown'},
                timeout=8)
        except Exception as e:
            log.warning(f"Pending signal notification failed: {e}")

    def _auto_execute(self, signals):
        if _extra_bool('approval_mode', False):
            self._notify_pending_signals(signals)
            log.info("Auto-execute SKIPPED - approval mode is ON.")
            return

        if getattr(self.config, 'pyramid_enabled', False):
            for signal in signals:
                try:
                    self._check_pyramid_opportunity(signal)
                except Exception as e:
                    log.debug(f"Pyramid check error: {e}")

        for signal in signals:
            action = signal.get('action')
            confidence = signal.get('confidence', 0)
            pair = signal.get('pair')
            if action not in ('buy', 'sell'):
                continue
            if confidence < AUTO_EXECUTE_MIN_CONFIDENCE:
                log.info(f"Auto-execute skipped {pair} - confidence {confidence}% below {AUTO_EXECUTE_MIN_CONFIDENCE}%")
                continue
            approved, reason = self.risk_manager.check_trade(pair, action, confidence)
            if not approved:
                log.info(f"Auto-execute blocked {pair}: {reason}")
                continue
            try:
                if action == 'sell':
                    hold_reason = self._check_min_hold(pair)
                    if hold_reason is not None:
                        log.info(f"Skipping auto-sell {pair} - {hold_reason}")
                        self.risk_manager.release_lock(pair)
                        continue

                if action == 'sell' and _extra_bool('trailing_stop_enabled', False):
                    try:
                        base = pair.replace('/USDT', '').replace('/', '')
                        sym = base + 'USDT'
                        entry = (self.high_water_marks.get(sym, {}).get('entry_price')
                                 if hasattr(self, 'high_water_marks') else None)
                        if entry:
                            cur = float(self.trader.client.get_symbol_ticker(symbol=sym)['price'])
                            gain_pct = ((cur - entry) / entry) * 100
                            if gain_pct >= 2.0:
                                log.info(f"Skipping auto-sell {pair} - up {gain_pct:.1f}%, trailing will handle")
                                self.risk_manager.release_lock(pair)
                                continue
                    except Exception:
                        pass

                if action == 'sell':
                    try:
                        if self.manual_manager and self.manual_manager.has_position(pair):
                            log.info(f"Skipping auto-sell {pair} - manual position")
                            self.risk_manager.release_lock(pair)
                            continue
                    except Exception:
                        pass

                if action == 'sell':
                    try:
                        base = pair.replace('/USDT', '')
                        symbol = pair.replace('/', '')
                        account = self.trader.client.get_account()
                        balance = next(
                            (float(b['free']) + float(b['locked'])
                             for b in account['balances'] if b['asset'] == base), 0.0
                        )
                        price = float(self.trader.client.get_symbol_ticker(symbol=symbol)['price'])
                        value = balance * price
                        if value < 2.0:
                            log.info(f"Skipping sell {pair} - no holdings (${value:.2f})")
                            self.risk_manager.release_lock(pair)
                            continue
                    except Exception:
                        self.risk_manager.release_lock(pair)
                        continue

                if action == 'buy':
                    try:
                        self.trader._cancel_all_open_orders(pair)
                    except Exception:
                        pass
                    self.risk_manager.record_trade(pair)
                result = self.trader.execute_trade(pair, action, self.config.max_trade_pct)
                self.risk_manager.record_trade(pair)
                if action == 'buy':
                    self._record_buy_time(pair)
                log.info(f"Auto-executed: {action.upper()} {pair} - confidence {confidence}%")
                try:
                    icon = '🟢' if action == 'buy' else '🔴'
                    msg = f"{icon} *AUTO {action.upper()} {pair}*\nConfidence: {confidence}%"
                    self._tg_send(msg, context=f"auto-{action}-{pair}")
                except Exception as te:
                    log.warning(f"Telegram block crashed: {te}")
            except Exception as e:
                log.error(f"Auto-execute failed for {pair}: {e}")
            finally:
                self.risk_manager.release_lock(pair)

    def _tg_send(self, message, context='generic', use_markdown=True):
        tok = getattr(self.config, 'telegram_token', '') or ''
        chat = getattr(self.config, 'telegram_chat_id', '') or ''
        if not tok or not chat:
            return False
        payload = {'chat_id': str(chat), 'text': message}
        if use_markdown:
            payload['parse_mode'] = 'Markdown'
        try:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", json=payload, timeout=10)
            if r.status_code == 200:
                log.info(f"Telegram sent ({context})")
                return True
            else:
                log.warning(f"Telegram returned {r.status_code} ({context}): {r.text[:300]}")
                return False
        except Exception as e:
            log.warning(f"Telegram exception ({context}): {e}")
            return False

    def get_latest_signals(self):
        if not self.latest_signals:
            self.refresh_signals()
        return self.latest_signals

    def analyse_pair(self, symbol):
        klines = self.trader.get_klines(symbol, '1h', 100)
        if len(klines) < 50:
            return None
        closes = [k['close'] for k in klines]
        volumes = [k['volume'] for k in klines]
        rsi = self._rsi(closes)
        macd = self._macd(closes)
        ma50 = self._sma(closes, 50)
        ma200 = self._sma(closes, 200) if len(closes) >= 200 else None
        bb = self._bollinger(closes)
        stoch_rsi = self._stochastic_rsi(closes)
        rsi_divergence = self._detect_rsi_divergence(closes)
        bb_bounce = self._detect_bb_bounce(closes, bb)
        volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if volumes else 1.0
        price = closes[-1]
        price_change = ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 else 0

        if hasattr(self.config, 'get_pair_rsi'):
            base_buy, base_sell = self.config.get_pair_rsi(symbol)
        else:
            base_buy, base_sell = self.config.rsi_buy, self.config.rsi_sell

        if self.market_regime == 'bearish':
            rsi_buy_threshold = base_buy - 3
            rsi_sell_threshold = base_sell - 5
        elif self.market_regime == 'bullish':
            rsi_buy_threshold = base_buy + 3
            rsi_sell_threshold = base_sell + 3
        else:
            rsi_buy_threshold = base_buy
            rsi_sell_threshold = base_sell

        indicators = {
            'rsi': rsi, 'macd': macd, 'ma50': ma50, 'ma200': ma200,
            'bb': bb, 'stoch_rsi': stoch_rsi, 'rsi_divergence': rsi_divergence,
            'bb_bounce': bb_bounce, 'volume_ratio': volume_ratio, 'price': price,
            'price_change_24h': price_change,
            'rsi_buy_threshold': rsi_buy_threshold,
            'rsi_sell_threshold': rsi_sell_threshold,
            'fear_greed': getattr(self, 'fear_greed_index', 50)
        }
        return self._ai_analyse(symbol, indicators)

    def _ai_analyse(self, symbol, indicators):
        """Hybrid signal generation:

        1. Score indicators deterministically -> baseline (action, confidence, factors).
           Confidence varies meaningfully per pair because it's derived from
           how many indicators align and how strong each is.
        2. Ask AI to confirm/override the action and write a narrative reason.
           AI does NOT set confidence - that's our deterministic value. This
           is the fix for "stuck at 70%": the LLM was anchoring on the
           placeholder value in the JSON example.
        3. If AI is unavailable or returns bad JSON, fall through to step 1's
           values with factors as the reason.
        """
        try:
            base_action, base_confidence, factors = self._score_indicators(indicators)

            # No AI keys at all -> use deterministic result directly
            if not (_openai_key() or _gemini_key()):
                return self._build_signal_from_score(symbol, indicators,
                                                     base_action, base_confidence, factors)

            rsi = indicators['rsi']
            macd = indicators['macd']
            ma50 = indicators['ma50']
            ma200 = indicators.get('ma200')
            price = indicators['price']
            bb = indicators.get('bb', {})
            volume_ratio = indicators.get('volume_ratio', 1.0)
            price_change = indicators['price_change_24h']
            ma200_str = str(round(ma200, 4)) if ma200 is not None else 'N/A'
            macd_signal = macd.get('signal', 'neutral')
            macd_hist = macd.get('histogram', 0)
            bb_pos = bb.get('position', 50)

            prompt = (
                f"You are reviewing a quant scoring engine's output for {symbol}.\n\n"
                f"The engine has scored the indicators and produced:\n"
                f"  Suggested action: {base_action}\n"
                f"  Confidence (deterministic, you CANNOT change this): {base_confidence}%\n"
                f"  Contributing factors: {' | '.join(factors[:6]) if factors else 'none'}\n\n"
                f"Live data:\n"
                f"  Market regime: {self.market_regime}, F&G: {indicators.get('fear_greed', 50)}/100\n"
                f"  RSI {round(rsi, 1)}, MACD {macd_signal} (hist {round(macd_hist, 4)})\n"
                f"  Price {round(price, 4)}, 50MA {round(ma50, 4)}, 200MA {ma200_str}\n"
                f"  BB position {round(bb_pos, 0)}%, Volume {round(volume_ratio, 1)}x avg, "
                f"24h change {round(price_change, 2)}%\n"
                f"  RSI buy<{indicators.get('rsi_buy_threshold')}, "
                f"sell>{indicators.get('rsi_sell_threshold')} (regime-adjusted)\n\n"
                f"YOUR JOB:\n"
                f"1. Confirm or override the action. Only override if you see something "
                f"the indicators clearly missed (e.g. a textbook divergence the scorer "
                f"didn't catch, or a regime-specific risk).\n"
                f"2. Write a clear 2-sentence reason that references the ACTUAL numbers above. "
                f"Don't be generic.\n\n"
                f"IMPORTANT RULES:\n"
                f"- TP/SL is handled by exchange OCO orders, NOT by you. Don't generate "
                f"sell signals on normal pullbacks. Prefer 'watch' over 'sell' if uncertain.\n"
                f"- A 4-hour minimum hold period blocks premature AI sells, so a sell "
                f"signal must be a genuine reversal, not noise.\n"
                f"- You CANNOT return a confidence number. It is set deterministically.\n\n"
                f"Return ONLY this JSON, no markdown:\n"
                f'{{"action":"buy|sell|watch|hold","reason":"two sentences citing real numbers"}}'
            )

            text = _call_ai(prompt, max_tokens=300, _err_state=self._ai_err_state)
            ai_data = _extract_json_block(text) if text else None

            action = base_action
            confidence = base_confidence
            reason = '. '.join(factors[:4]) + '.' if factors else 'No strong signal.'

            if ai_data and isinstance(ai_data, dict):
                ai_action = str(ai_data.get('action', '')).lower().strip()
                if ai_action in ('buy', 'sell', 'watch', 'hold'):
                    action = ai_action
                ai_reason = str(ai_data.get('reason', '')).strip()
                if ai_reason and len(ai_reason) > 10:
                    reason = ai_reason

            # If AI changed the action significantly, adjust confidence
            # to reflect the disagreement rather than blindly using base_confidence.
            if action != base_action:
                if action in ('watch', 'hold'):
                    # AI being more cautious than the score -> dampen confidence
                    confidence = max(50, base_confidence - 15)
                elif (action == 'buy' and base_action == 'sell') or \
                     (action == 'sell' and base_action == 'buy'):
                    # AI strongly disagrees with the score -> low confidence either way
                    confidence = 55

            return {
                'pair': symbol.replace('USDT', '') + '/USDT',
                'action': action,
                'confidence': int(confidence),
                'reason': reason,
                'rsi': round(indicators['rsi']),
                'macd': indicators['macd'].get('signal', 'neutral'),
                'trend': 'up' if indicators['price'] > indicators['ma50'] else 'down'
            }
        except Exception as e:
            log.warning(f"AI analysis failed for {symbol}: {e}")
            return self._fallback_signal(symbol, indicators)

    def _score_indicators(self, indicators):
        """Deterministic indicator scoring. Returns (action, confidence, factors).

        Score is a signed value: positive = bullish, negative = bearish.
        Magnitude reflects strength of conviction. Each indicator contributes
        an amount proportional to how strong its signal is, so different pairs
        with different setups produce meaningfully different confidence values
        instead of all clustering around an LLM's anchored default.
        """
        rsi = indicators['rsi']
        macd = indicators['macd']
        price = indicators['price']
        ma50 = indicators['ma50']
        bb = indicators.get('bb', {})
        volume_ratio = indicators.get('volume_ratio', 1.0)
        stoch_rsi = indicators.get('stoch_rsi', 50)
        rsi_div = indicators.get('rsi_divergence', 'none')
        bb_bounce = indicators.get('bb_bounce', False)
        rsi_buy = indicators.get('rsi_buy_threshold', getattr(self.config, 'rsi_buy', 30))
        rsi_sell = indicators.get('rsi_sell_threshold', getattr(self.config, 'rsi_sell', 70))
        fg = indicators.get('fear_greed', 50)

        score = 0.0
        factors = []

        # RSI - scaled by distance past threshold
        if rsi < rsi_buy:
            gain = min((rsi_buy - rsi) * 1.2, 22)
            score += gain
            factors.append(f"RSI oversold {rsi:.1f}")
        elif rsi > rsi_sell:
            loss = min((rsi - rsi_sell) * 1.2, 22)
            score -= loss
            factors.append(f"RSI overbought {rsi:.1f}")
        elif rsi < 40:
            score += 4
        elif rsi > 60:
            score -= 4

        # MACD - histogram magnitude as % of price (so it's comparable
        # between BTC at $77k and DOGE at $0.10)
        macd_sig = macd.get('signal', 'neutral')
        macd_hist = abs(macd.get('histogram', 0))
        macd_strength_pct = (macd_hist / price * 100) if price > 0 else 0
        if macd_sig == 'bullish':
            w = 8 + min(macd_strength_pct * 4, 5)
            score += w
            factors.append(f"MACD bullish")
        elif macd_sig == 'bearish':
            w = 8 + min(macd_strength_pct * 4, 5)
            score -= w
            factors.append(f"MACD bearish")

        # Price vs 50MA - trend filter
        if ma50 and ma50 > 0:
            pct_off = (price - ma50) / ma50 * 100
            if pct_off > 2:
                score += 8
                factors.append(f"+{pct_off:.1f}% over 50MA")
            elif pct_off < -2:
                score -= 8
                factors.append(f"{pct_off:.1f}% under 50MA")

        # Bollinger bands
        if bb_bounce:
            score += 12
            factors.append("BB lower bounce")
        else:
            bb_position = bb.get('position', 50) if bb else 50
            if bb_position > 90:
                score -= 8
                factors.append("Above upper BB")
            elif bb_position < 10:
                score += 6
                factors.append("Near lower BB")

        # RSI divergence - strongest single signal
        if rsi_div == 'bullish':
            score += 15
            factors.append("Bullish RSI divergence")
        elif rsi_div == 'bearish':
            score -= 15
            factors.append("Bearish RSI divergence")

        # Stochastic RSI
        if stoch_rsi < 20:
            score += 7
            factors.append(f"Stoch RSI {stoch_rsi:.0f}")
        elif stoch_rsi > 80:
            score -= 7
            factors.append(f"Stoch RSI {stoch_rsi:.0f}")

        # Volume - amplifies existing direction
        if volume_ratio > 1.5:
            if score > 0:
                score += 5
                factors.append(f"Vol {volume_ratio:.1f}x confirms")
            elif score < 0:
                score -= 5
                factors.append(f"Vol {volume_ratio:.1f}x confirms")
        elif volume_ratio < 0.7:
            score *= 0.85
            factors.append(f"Low vol {volume_ratio:.1f}x")

        # Fear & Greed contrarian boost on extremes
        if fg < 20 and score > 0:
            score += 5
            factors.append(f"F&G {fg} extreme fear")
        elif fg > 80 and score < 0:
            score -= 5
            factors.append(f"F&G {fg} extreme greed")

        # Regime adjustment - dampen counter-regime signals
        if self.market_regime == 'bearish' and score > 0:
            score *= 0.85
        elif self.market_regime == 'bullish' and score < 0:
            score *= 0.85

        # Score -> action
        if score >= 18:
            action = 'buy'
        elif score <= -18:
            action = 'sell'
        elif abs(score) >= 6:
            action = 'watch'
        else:
            action = 'hold'

        # Score -> confidence (50-95 range)
        confidence = int(min(50 + abs(score) * 1.4, 95))

        return action, confidence, factors

    def _build_signal_from_score(self, symbol, indicators, action, confidence, factors):
        """Build a signal dict purely from deterministic scoring. Used when no
        AI provider is configured."""
        rsi = indicators['rsi']
        macd = indicators['macd']
        price = indicators['price']
        ma50 = indicators['ma50']
        reason = '. '.join(factors[:4]) + '.' if factors else 'No strong signal.'
        return {
            'pair': symbol.replace('USDT', '') + '/USDT',
            'action': action,
            'confidence': int(confidence),
            'reason': reason,
            'rsi': round(rsi),
            'macd': macd.get('signal', 'neutral'),
            'trend': 'up' if price > ma50 else 'down'
        }

    def _fallback_signal(self, symbol, indicators):
        rsi = indicators['rsi']
        macd = indicators['macd']
        price = indicators['price']
        ma50 = indicators['ma50']
        macd_signal = macd.get('signal', 'neutral')
        rsi_buy_threshold = indicators.get('rsi_buy_threshold', self.config.rsi_buy)
        rsi_sell_threshold = indicators.get('rsi_sell_threshold', self.config.rsi_sell)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        bb = indicators.get('bb', {})
        action = 'hold'
        confidence = 50
        reasons = []

        if rsi < rsi_buy_threshold:
            reasons.append("RSI oversold at " + str(round(rsi, 1)))
            fg = indicators.get('fear_greed', 50)
            if fg < 15:
                action = 'hold'; confidence = 35
            elif ma50 > 0 and price < ma50 * 0.985:
                action = 'hold'; confidence = 40
            elif volume_ratio < 0.8:
                action = 'hold'; confidence = 45
            else:
                action = 'buy'
                confidence += 15
                bb_bounce = indicators.get('bb_bounce', False)
                if bb_bounce:
                    confidence += 12
                    reasons.append("Bouncing off lower BB")
                rsi_div = indicators.get('rsi_divergence', 'none')
                if rsi_div == 'bullish':
                    confidence += 15
                    reasons.append("Bullish RSI divergence")
                stoch = indicators.get('stoch_rsi', 50)
                if stoch < 20:
                    confidence += 10
                if price > ma50:
                    confidence += 8
                if volume_ratio > 1.5:
                    confidence += 8
        elif rsi > rsi_sell_threshold:
            reasons.append("RSI overbought at " + str(round(rsi, 1)))
            action = 'sell'
            confidence += 15

        if macd_signal == 'bullish' and action != 'sell':
            reasons.append("MACD bullish crossover")
            action = 'buy'
            confidence += 10
        elif macd_signal == 'bearish' and action != 'buy':
            reasons.append("MACD bearish crossover")
            action = 'sell'
            confidence += 10

        if price > ma50 and action == 'buy':
            confidence += 10
        elif price < ma50 and action == 'sell':
            confidence += 10

        if not reasons:
            action = 'watch'

        return {
            'pair': symbol.replace('USDT', '') + '/USDT',
            'action': action,
            'confidence': min(confidence, 95),
            'reason': '. '.join(reasons) + '.' if reasons else 'No strong signal.',
            'rsi': round(rsi),
            'macd': macd_signal,
            'trend': 'up' if price > ma50 else 'down'
        }

    def _rsi(self, closes, period=14):
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

    def _sma(self, closes, period):
        if len(closes) < period:
            return closes[-1]
        return float(np.mean(closes[-period:]))

    def _ema(self, closes, period):
        if len(closes) < period:
            return closes[-1]
        k = 2 / (period + 1)
        ema = closes[0]
        for p in closes[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def _ema_series(self, closes, period):
        """Return the full EMA series, not just the final value. Needed by
        MACD signal line which is EMA(9) OF THE MACD LINE, not of closes."""
        if len(closes) < period:
            return [closes[-1]] if closes else []
        k = 2 / (period + 1)
        series = [closes[0]]
        for p in closes[1:]:
            series.append(p * k + series[-1] * (1 - k))
        return series

    def _macd(self, closes):
        """Standard MACD:
          MACD line  = EMA(12, closes) - EMA(26, closes)        # series
          Signal     = EMA(9, MACD line)                         # series
          Histogram  = MACD line - Signal line

        The previous implementation computed `signal_line = EMA(9, closes)`
        which for high-priced assets like BTC made the signal line ≈ price
        itself, giving nonsense histograms (e.g. -76610) and biasing the
        cross to always 'bearish'. Fixed.
        """
        if len(closes) < 26 + 9:
            return {'signal': 'neutral', 'histogram': 0}
        ema12 = self._ema_series(closes, 12)
        ema26 = self._ema_series(closes, 26)
        macd_line_series = [a - b for a, b in zip(ema12, ema26)]
        signal_line_series = self._ema_series(macd_line_series, 9)
        macd_line = macd_line_series[-1]
        signal_line = signal_line_series[-1]
        histogram = macd_line - signal_line
        if macd_line > signal_line:
            signal = 'bullish'
        elif macd_line < signal_line:
            signal = 'bearish'
        else:
            signal = 'neutral'
        return {'signal': signal, 'histogram': histogram}

    def _stochastic_rsi(self, closes, rsi_period=14, stoch_period=14):
        try:
            if len(closes) < rsi_period + stoch_period:
                return 50.0
            rsi_values = []
            for i in range(rsi_period, len(closes)):
                rsi_values.append(self._rsi(closes[:i+1]))
            if len(rsi_values) < stoch_period:
                return 50.0
            recent_rsi = rsi_values[-stoch_period:]
            min_rsi = min(recent_rsi)
            max_rsi = max(recent_rsi)
            if max_rsi == min_rsi:
                return 50.0
            stoch = ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100
            return round(stoch, 2)
        except Exception:
            return 50.0

    def _detect_rsi_divergence(self, closes, lookback=10):
        try:
            if len(closes) < lookback + 14:
                return 'none'
            recent_closes = closes[-(lookback+14):]
            rsi_now = self._rsi(recent_closes)
            rsi_prev = self._rsi(recent_closes[:-5])
            price_now = closes[-1]
            price_prev = min(closes[-(lookback):-5])
            if price_now < price_prev and rsi_now > rsi_prev + 2:
                return 'bullish'
            if price_now > price_prev and rsi_now < rsi_prev - 2:
                return 'bearish'
            return 'none'
        except Exception:
            return 'none'

    def _detect_bb_bounce(self, closes, bb):
        try:
            if not bb or 'lower' not in bb:
                return False
            lower = bb['lower']
            price = closes[-1]
            prev_price = closes[-2] if len(closes) >= 2 else price
            near_lower = price < lower * 1.015
            bouncing = price > prev_price
            return bool(near_lower and bouncing)
        except Exception:
            return False

    def _bollinger(self, closes, period=20):
        if len(closes) < period:
            return {'position': 50}
        recent = closes[-period:]
        mid = np.mean(recent)
        std = np.std(recent)
        upper = mid + 2 * std
        lower = mid - 2 * std
        band_range = upper - lower
        position = ((closes[-1] - lower) / band_range * 100) if band_range > 0 else 50
        return {'upper': upper, 'lower': lower, 'mid': mid, 'position': position}
