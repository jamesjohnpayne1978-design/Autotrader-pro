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
from config import Config

log = logging.getLogger(__name__)

# ============================================================
# AI PROVIDER ENDPOINTS AND MODELS
# ============================================================
OPENAI_API   = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"   # cheap, fast, well-suited to structured JSON output

GEMINI_API   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
# ============================================================


# ============================================================
# AUTO-EXECUTE CONFIDENCE THRESHOLDS - tune these to taste
# ============================================================
# Lower = more aggressive (more trades, more false signals)
# Higher = more conservative (fewer trades, higher win rate)
# These ONLY apply when Auto Mode is ON. Approval mode is unaffected.
AUTO_EXECUTE_MIN_CONFIDENCE = 60
PYRAMID_MIN_CONFIDENCE      = 65
# ============================================================


def _openai_key():
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _gemini_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _call_ai(prompt, max_tokens=500, _err_state=None):
    """Call AI to analyse a prompt. Tries OpenAI first, then Gemini.

    Returns the raw text response from whichever provider succeeded, or None
    if both failed (or neither is configured). Caller is responsible for
    JSON parsing.
    """
    # ---- OpenAI (primary) ----
    okey = _openai_key()
    if okey:
        try:
            r = requests.post(
                OPENAI_API,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {okey}",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.4,
                },
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                _log_ai_error("OpenAI", r.status_code, r.text, _err_state)
        except Exception as e:
            _log_ai_error("OpenAI", "exc", str(e), _err_state)

    # ---- Gemini (fallback) ----
    gkey = _gemini_key()
    if gkey:
        try:
            r = requests.post(
                f"{GEMINI_API}?key={gkey}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": max_tokens,
                    },
                },
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Strip markdown code fences Gemini sometimes adds
                text = text.replace("```json", "").replace("```", "").strip()
                return text
            else:
                _log_ai_error("Gemini", r.status_code, r.text, _err_state)
        except Exception as e:
            _log_ai_error("Gemini", "exc", str(e), _err_state)

    return None


def _log_ai_error(provider, status, body, state):
    """Log AI errors verbosely the first 3 times, then suppress to avoid log spam."""
    if state is None:
        return
    state["count"] = state.get("count", 0) + 1
    if state["count"] <= 3:
        msg = body[:200] if isinstance(body, str) else body
        log.warning(f"AI provider {provider} failed (status={status}): {msg}")
    elif state["count"] == 4:
        log.warning(f"AI provider {provider} continuing to fail - suppressing further error logs")


def _extract_json_block(text):
    """Find and parse the first {...} JSON object in text. Returns None if none."""
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

        # Shared error state so we can rate-limit log spam if AI is down
        self._ai_err_state = {"count": 0}

        # Make AI status visible at startup so we never silently run on fallback again
        openai = bool(_openai_key())
        gemini = bool(_gemini_key())
        if openai and gemini:
            log.info(f"AI providers configured: OpenAI/{OPENAI_MODEL} (primary) + Gemini (fallback)")
        elif openai:
            log.info(f"AI provider configured: OpenAI/{OPENAI_MODEL}")
        elif gemini:
            log.info("AI provider configured: Gemini (free tier)")
        else:
            log.warning("NO AI PROVIDER CONFIGURED - all signals will use rule-based fallback. "
                        "Set OPENAI_API_KEY or GEMINI_API_KEY in Railway Variables.")

    def run(self):
        self.running = True
        log.info("Signal engine started.")
        while self.running:
            # Fetch Fear & Greed every 30 mins (every 6 signal cycles)
            if not hasattr(self, '_fg_count'):
                self._fg_count = 0
            self._fg_count += 1
            if self._fg_count % 6 == 1:
                self._fetch_fear_greed()

            try:
                self.detect_market_regime()
                self.refresh_signals()
                # Check trailing stops every cycle
                try:
                    if self.trader:
                        self.trader.check_all_trailing_stops()
                except Exception as e:
                    log.debug(f"Trailing stop check error: {e}")
                # Run our high-level trailing stop monitor (separate from
                # trader.py's stop logic - works on the portfolio level)
                try:
                    self._check_portfolio_trailing_stops()
                except Exception as e:
                    log.debug(f"Portfolio trailing stop error: {e}")
                # Check for over-concentration in any single position
                try:
                    self._check_position_concentration()
                except Exception as e:
                    log.debug(f"Concentration check error: {e}")
            except Exception as e:
                log.error(f"Signal refresh error: {e}")
            time.sleep(300)

    def _check_portfolio_trailing_stops(self):
        """Portfolio-level trailing stop loss. Tracks the high-water mark for
        every open position. When price drops more than `trailing_stop_pct`
        from the peak (and we're already in profit by `trailing_stop_activate_pct`),
        triggers a sell.

        This runs independently of trader.py's own stop-loss logic. Both can fire;
        whichever fires first wins, the second one is a no-op.

        Disabled by default - enable via config.trailing_stop_enabled = True.
        """
        if not getattr(self.config, 'trailing_stop_enabled', False):
            return
        if not self.trader:
            return

        # Init high-water mark tracker
        if not hasattr(self, 'high_water_marks'):
            self.high_water_marks = {}  # {symbol: {entry_price, peak_price}}

        trail_pct = float(getattr(self.config, 'trailing_stop_pct', 3.0))
        activate_pct = float(getattr(self.config, 'trailing_stop_activate_pct', 2.0))

        try:
            account = self.trader.client.get_account()
            prices = self.trader.client.get_all_tickers()
            price_map = {p['symbol']: float(p['price']) for p in prices}

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
                if value < 5.0:  # Skip dust
                    continue

                # Initialise entry if first time we've seen this position
                if symbol not in self.high_water_marks:
                    self.high_water_marks[symbol] = {
                        'entry_price': current_price,
                        'peak_price': current_price,
                    }
                    continue  # Wait one cycle before tracking

                hwm = self.high_water_marks[symbol]
                # Update peak
                if current_price > hwm['peak_price']:
                    hwm['peak_price'] = current_price

                # Calculate position state
                gain_from_entry = ((current_price - hwm['entry_price']) / hwm['entry_price']) * 100
                drop_from_peak = ((hwm['peak_price'] - current_price) / hwm['peak_price']) * 100

                # Don't trail unless we're up by the activation threshold
                if gain_from_entry < activate_pct:
                    continue

                # Skip if it's a manual position - manual manager handles those
                pair_slash = symbol.replace('USDT', '/USDT')
                try:
                    if self.manual_manager and self.manual_manager.has_position(pair_slash):
                        continue
                except Exception:
                    pass

                # Trigger sell if dropped from peak by trailing percentage
                if drop_from_peak >= trail_pct:
                    log.info(f"TRAILING STOP triggered for {symbol}: peak ${hwm['peak_price']:.4f}, "
                             f"current ${current_price:.4f} (drop {drop_from_peak:.1f}%), "
                             f"locked-in gain {gain_from_entry:.1f}%")
                    try:
                        if self.risk_manager:
                            approved, _ = self.risk_manager.check_trade(pair_slash, 'sell', 100)
                            if not approved:
                                continue
                        result = self.trader.execute_trade(pair_slash, 'sell', 100.0)  # Sell all
                        log.info(f"Trailing stop SELL executed for {pair_slash}: {result}")
                        # Clear tracker so next entry starts fresh
                        del self.high_water_marks[symbol]
                        # Telegram alert
                        try:
                            if self.config.telegram_token and self.config.telegram_chat_id:
                                msg = (
                                    f"🎯 *TRAILING STOP SELL {pair_slash}*\n"
                                    f"Locked in {gain_from_entry:.1f}% gain\n"
                                    f"Peak: ${hwm['peak_price']:.4f}\n"
                                    f"Exit: ${current_price:.4f} ({drop_from_peak:.1f}% from peak)"
                                )
                                requests.post(
                                    f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                                    json={'chat_id': str(self.config.telegram_chat_id), 'text': msg, 'parse_mode': 'Markdown'},
                                    timeout=8
                                )
                        except Exception:
                            pass
                        if self.risk_manager:
                            self.risk_manager.release_lock(pair_slash)
                    except Exception as e:
                        log.error(f"Trailing stop sell failed for {pair_slash}: {e}")
                        if self.risk_manager:
                            try:
                                self.risk_manager.release_lock(pair_slash)
                            except Exception:
                                pass
        except Exception as e:
            log.debug(f"Portfolio trailing stop scan error: {e}")

    def _check_position_concentration(self):
        """Send a Telegram alert when any single position exceeds 25% of the
        total portfolio value. Helps catch over-concentration risk before a
        single bad trade tanks the whole portfolio.

        Alerts are throttled to once every 4 hours per pair to avoid spam.
        """
        if not self.trader:
            return

        threshold_pct = float(getattr(self.config, 'concentration_alert_pct', 25.0))
        cooldown_hours = 4

        if not hasattr(self, '_last_concentration_alert'):
            self._last_concentration_alert = {}

        try:
            account = self.trader.client.get_account()
            prices = self.trader.client.get_all_tickers()
            price_map = {p['symbol']: float(p['price']) for p in prices}

            # Calculate value of every non-stable balance
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
                return  # Too small to bother

            # Find over-concentrated positions
            now = time.time()
            for asset, value in holdings.items():
                pct = (value / total_value) * 100
                if pct < threshold_pct:
                    continue
                # Check cooldown
                last_alert = self._last_concentration_alert.get(asset, 0)
                if now - last_alert < cooldown_hours * 3600:
                    continue

                log.info(f"Concentration alert: {asset} is {pct:.1f}% of portfolio (${value:.2f} / ${total_value:.2f})")
                self._last_concentration_alert[asset] = now

                # Send Telegram
                try:
                    if self.config.telegram_token and self.config.telegram_chat_id:
                        msg = (
                            f"⚖️ *CONCENTRATION ALERT*\n"
                            f"{asset} is *{pct:.1f}%* of your portfolio (${value:.2f}).\n"
                            f"Total portfolio: ${total_value:.2f}\n"
                            f"Consider taking partial profit to rebalance."
                        )
                        requests.post(
                            f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                            json={'chat_id': str(self.config.telegram_chat_id), 'text': msg, 'parse_mode': 'Markdown'},
                            timeout=8
                        )
                except Exception:
                    pass
        except Exception as e:
            log.debug(f"Concentration check error: {e}")

    def _fetch_fear_greed(self):
        """Fetch crypto Fear & Greed index - 0=extreme fear, 100=extreme greed"""
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
        """Analyse overall market to determine bull/bear/neutral regime"""
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
                f"Volume trend (3d vs 14d avg): {round(volume_trend, 2)}x\n"
                f"Crypto Fear & Greed Index: {self.fear_greed_index}/100 ({self.fear_greed_label})\n\n"
                f"Determine:\n"
                f"1. Market regime: bullish, bearish, or neutral\n"
                f"2. Recommended take profit %: bullish=15, neutral=6, bearish=8\n\n"
                f"Return ONLY valid JSON, no markdown, no explanation:\n"
                f'{{"regime":"bullish|bearish|neutral","take_profit":6,"reason":"one sentence explanation"}}'
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

            # AI unavailable or returned bad output - use rule-based fallback
            self._fallback_regime(rsi, price_7d_change, price, ma7)

        except Exception as e:
            log.warning(f"Regime detection failed: {e}")
            self._fallback_regime_simple()

    def _fallback_regime(self, rsi, price_change_7d, price, ma7):
        """Rule-based regime detection when AI unavailable"""
        if rsi > 55 and price_change_7d > 3 and price > ma7:
            self.market_regime = 'bullish'
            self.regime_tp = 15.0
            self.regime_reason = f"RSI {round(rsi)} above 55, price up {round(price_change_7d, 1)}% in 7 days and above 7MA."
        elif rsi < 45 and price_change_7d < -3 and price < ma7:
            self.market_regime = 'bearish'
            self.regime_tp = 8.0
            self.regime_reason = f"RSI {round(rsi)} below 45, price down {round(abs(price_change_7d), 1)}% in 7 days and below 7MA."
        else:
            self.market_regime = 'neutral'
            self.regime_tp = 6.0
            self.regime_reason = f"Mixed signals - RSI {round(rsi)}, 7-day change {round(price_change_7d, 1)}%."

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
        return {
            'regime': self.market_regime,
            'take_profit': self.regime_tp,
            'reason': self.regime_reason
        }

    def refresh_signals(self):
        # Belt-and-suspenders: release any locks left behind by a crashed
        # auto-execute thread from the previous cycle. The cycle thread is
        # single-threaded so by the time refresh runs, no legitimate lock
        # should still be held. If one is, it's stale and must be cleared.
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
        log.info(f"Signals refreshed: {len(signals)} signals - regime: {self.market_regime} - TP: {self.regime_tp}%")
        if self.config.auto_mode and self.risk_manager:
            self._auto_execute(signals)

    def _check_pyramid_opportunity(self, signal):
        """Independent pyramid check that runs for EVERY pair we hold,
        regardless of what action the AI suggested. Pyramid is fundamentally
        a 'price dropped from my last buy' decision, not an 'AI says buy
        right now' decision. This decouples the two.

        Conditions for pyramid to fire:
        - AI action is NOT 'sell' (don't add to a position the AI wants out of)
        - AI confidence on the signal is >= PYRAMID_MIN_CONFIDENCE
          (so we don't add when the AI thinks the asset is junk)
        - trader.should_pyramid() approves: typically price dropped >=
          pyramid_drop_trigger% from last buy AND haven't exceeded max adds
          AND not below max drawdown
        - Risk manager approves (cooldown elapsed, daily loss limit OK)
        """
        pair = signal.get('pair')
        action = signal.get('action', 'hold')
        confidence = signal.get('confidence', 0)

        # Don't pyramid if AI actively wants out of this pair
        if action == 'sell':
            return

        # Don't pyramid if AI confidence is too low - we'd be adding to junk
        if confidence < PYRAMID_MIN_CONFIDENCE:
            return

        # Don't pyramid manual positions - they have their own TP/SL logic
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
            log.debug(f"Pyramid should_add check failed for {pair}: {e}")
            return

        if not should_add:
            return

        # should_pyramid() returned True - go through risk manager
        approved, rm_reason = self.risk_manager.check_trade(pair, 'buy', confidence)
        if not approved:
            log.info(f"Pyramid {pair} blocked by risk manager: {rm_reason}")
            try:
                self.risk_manager.release_lock(pair)
            except Exception:
                pass
            return

        try:
            log.info(f"Pyramid opportunity for {pair} (AI action={action}, conf={confidence}): {reason}")
            pyramid_pct = getattr(self.config, 'pyramid_size_pct', 3.0)
            result = self.trader.execute_trade(pair, 'buy', pyramid_pct)
            self.risk_manager.record_trade(pair)
            log.info(f"Pyramid buy executed for {pair}: {result}")

            # Send Telegram notification
            try:
                if self.config.telegram_token and self.config.telegram_chat_id:
                    msg = (
                        f"🔼 *PYRAMID BUY {pair}*\n"
                        f"AI action: {action} (conf {confidence}%)\n"
                        f"Reason: {reason}\n"
                        f"Size: {pyramid_pct}% of portfolio"
                    )
                    requests.post(
                        f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                        json={'chat_id': str(self.config.telegram_chat_id), 'text': msg, 'parse_mode': 'Markdown'},
                        timeout=8
                    )
            except Exception:
                pass
        except Exception as e:
            log.error(f"Pyramid execute failed for {pair}: {e}")
        finally:
            try:
                self.risk_manager.release_lock(pair)
            except Exception:
                pass

    def _auto_execute(self, signals):
        # FIRST PASS: Check pyramid opportunities for ALL pairs we hold,
        # regardless of the AI's buy/sell action. Pyramid logic is about
        # "price dropped from last buy" not "AI said buy this second".
        if getattr(self.config, 'pyramid_enabled', False):
            for signal in signals:
                try:
                    self._check_pyramid_opportunity(signal)
                except Exception as e:
                    log.debug(f"Pyramid check error: {e}")

        # SECOND PASS: Normal buy/sell auto-execute on AI signals
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
                # NEVER auto-sell manual positions - they have their own TP/SL manager
                if action == 'sell':
                    try:
                        if self.manual_manager and self.manual_manager.has_position(pair):
                            log.info(f"Skipping auto-sell {pair} - open manual position (managed by TP/SL monitor)")
                            self.risk_manager.release_lock(pair)
                            continue
                    except Exception:
                        pass

                # Pyramid was already considered in first pass above - skip here

                # Skip sell if we have no holdings
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
                            log.info(f"Skipping sell {pair} - no meaningful holdings (value: ${value:.2f})")
                            self.risk_manager.release_lock(pair)
                            continue
                    except Exception as e:
                        log.debug(f"Balance check error for {pair}: {e}")
                        self.risk_manager.release_lock(pair)
                        continue

                # Cancel any stale open orders before buying to free up balance
                if action == 'buy':
                    try:
                        self.trader._cancel_all_open_orders(pair)
                    except Exception:
                        pass
                    self.risk_manager.record_trade(pair)
                result = self.trader.execute_trade(pair, action, self.config.max_trade_pct)
                self.risk_manager.record_trade(pair)  # Start cooldown
                log.info(f"Auto-executed: {action.upper()} {pair} - confidence {confidence}% - {result}")
                # Send Telegram notification for auto trades
                try:
                    if self.config.telegram_token and self.config.telegram_chat_id:
                        icon = '🟢' if action == 'buy' else '🔴'
                        tp = getattr(self.config, 'dynamic_tp', self.config.default_tp_pct)
                        msg = (
                            f"{icon} *AUTO {action.upper()} {pair}*\n"
                            f"Confidence: {confidence}%\n"
                            f"Market: {self.market_regime.upper()}\n"
                            f"TP: {tp}% · SL: {self.config.default_sl_pct}%"
                        )
                        requests.post(
                            f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                            json={'chat_id': str(self.config.telegram_chat_id), 'text': msg, 'parse_mode': 'Markdown'},
                            timeout=8
                        )
                except Exception as te:
                    log.debug(f"Telegram notification error: {te}")
            except Exception as e:
                log.error(f"Auto-execute failed for {pair}: {e}")
            finally:
                self.risk_manager.release_lock(pair)

    def get_latest_signals(self):
        if not self.latest_signals:
            self.refresh_signals()
            try:
                if self.trader:
                    self.trader.check_all_trailing_stops()
            except Exception as e:
                log.debug(f"Trailing stop check error: {e}")
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
        try:
            rsi = indicators['rsi']
            macd = indicators['macd']
            ma50 = indicators['ma50']
            ma200 = indicators['ma200']
            price = indicators['price']
            bb = indicators['bb']
            volume_ratio = indicators['volume_ratio']
            price_change = indicators['price_change_24h']
            ma200_str = str(round(ma200, 4)) if ma200 is not None else 'N/A'
            macd_signal = macd.get('signal', 'neutral')
            macd_hist = macd.get('histogram', 0)
            bb_pos = bb.get('position', 50)

            prompt = (
                f"Analyse {symbol} for trading signal. Market regime: {self.market_regime}.\n"
                f"Fear & Greed Index: {indicators.get('fear_greed', 50)}/100 "
                f"({self.fear_greed_label}) - extreme values often signal reversals.\n"
                f"RSI: {round(rsi, 1)}, MACD: {macd_signal} ({round(macd_hist, 4)})\n"
                f"Price: {round(price, 4)}, 50MA: {round(ma50, 4)}, 200MA: {ma200_str}\n"
                f"Above 50MA: {price > ma50}, BB position: {round(bb_pos, 0)}%\n"
                f"Volume ratio: {round(volume_ratio, 1)}x, 24h change: {round(price_change, 2)}%\n"
                f"RSI buy threshold: {indicators.get('rsi_buy_threshold')}, "
                f"sell threshold: {indicators.get('rsi_sell_threshold')} "
                f"(auto-adjusted for {self.market_regime} regime)\n"
                f"Recommended TP for this regime: {self.regime_tp}%\n\n"
                f"Decide whether to buy, sell, watch, or hold. Use confidence 0-100 to express conviction. "
                f"Higher confidence = stronger signal with multiple confirming indicators.\n"
                f"Return ONLY valid JSON, no markdown, no explanation:\n"
                f'{{"action":"buy|sell|watch|hold","confidence":75,"reason":"2-3 sentences",'
                f'"rsi":{round(rsi)},"macd":"bullish|bearish|neutral","trend":"up|down|sideways"}}'
            )

            text = _call_ai(prompt, max_tokens=500, _err_state=self._ai_err_state)
            signal_data = _extract_json_block(text) if text else None

            if signal_data and 'action' in signal_data:
                signal_data['pair'] = symbol.replace('USDT', '') + '/USDT'
                return signal_data

            return self._fallback_signal(symbol, indicators)

        except Exception as e:
            log.warning(f"AI analysis failed for {symbol}: {e}")
            return self._fallback_signal(symbol, indicators)

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
                log.info(f"Fear & Greed {fg} - Extreme Fear, skipping buy {symbol}")
                action = 'hold'
                confidence = 35
            elif ma50 > 0 and price < ma50 * 0.985:
                log.info(f"Trend filter: {symbol} downtrend, skipping buy")
                action = 'hold'
                confidence = 40
            elif volume_ratio < 0.8:
                log.info(f"Volume filter: {symbol} low volume ({volume_ratio:.1f}x), skipping")
                action = 'hold'
                confidence = 45
            else:
                action = 'buy'
                confidence += 15

                bb_bounce = indicators.get('bb_bounce', False)
                if bb_bounce:
                    confidence += 12
                    reasons.append("Price bouncing off lower Bollinger Band")
                elif bb and bb.get('position', 50) > 40:
                    confidence -= 8

                rsi_div = indicators.get('rsi_divergence', 'none')
                if rsi_div == 'bullish':
                    confidence += 15
                    reasons.append("Bullish RSI divergence detected")

                stoch = indicators.get('stoch_rsi', 50)
                if stoch < 20:
                    confidence += 10
                    reasons.append("Stochastic RSI oversold")
                elif stoch > 50:
                    confidence -= 5

                if price > ma50:
                    confidence += 8
                if volume_ratio > 1.5:
                    confidence += 8
                if fg > 40:
                    confidence += 5
                if fg < 25:
                    confidence -= 5
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
            'reason': '. '.join(reasons) + '.' if reasons else 'No strong signal. Monitoring.',
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

    def _macd(self, closes):
        if len(closes) < 26:
            return {'signal': 'neutral', 'histogram': 0}
        macd_line = self._ema(closes, 12) - self._ema(closes, 26)
        signal_line = self._ema(closes[-9:], 9)
        histogram = macd_line - signal_line
        signal = 'bullish' if macd_line > signal_line else 'bearish' if macd_line < signal_line else 'neutral'
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
