"""
AutoTrader Pro - AI Signal Engine
Added: Market regime detection + dynamic take profit adjustment
"""

import time
import logging
import json
import numpy as np
import requests
from config import Config

log = logging.getLogger(__name__)
ANTHROPIC_API = "https://api.anthropic.com/v1/messages"


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
            except Exception as e:
                log.error(f"Signal refresh error: {e}")
            time.sleep(300)

    def _fetch_fear_greed(self):
        """Fetch crypto Fear & Greed index - 0=extreme fear, 100=extreme greed"""
        try:
            import requests as req
            r = req.get('https://api.alternative.me/fng/?limit=1', timeout=8)
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
                f"You are a crypto market analyst. Determine the current market regime based on these Bitcoin indicators:\n\n"
                f"Current price: ${round(price, 0)}\n"
                f"7-day price change: {round(price_7d_change, 2)}%\n"
                f"RSI (14): {round(rsi, 1)}\n"
                f"Price vs 7-day MA: {'above' if price > ma7 else 'below'}\n"
                f"Price vs 14-day MA: {'above' if price > ma14 else 'below'}\n"
                f"Volume trend (3d vs 14d avg): {round(volume_trend, 2)}x\n\n"
                f"Based on these indicators, determine:\n"
                f"1. Market regime: bullish, bearish, or neutral\n"
                f"2. Recommended take profit %:\n"
                f"   - bullish = 8%\n"
                f"   - neutral = 6%\n"
                f"   - bearish = 4%\n\n"
                f"Return ONLY this JSON:\n"
                f'{{"regime":"bullish"|"bearish"|"neutral","take_profit":8|12|15,"reason":"one sentence explanation"}}'
            )

            response = requests.post(
                ANTHROPIC_API,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                text = data['content'][0]['text'].strip()
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1:
                    result = json.loads(text[start:end])
                    self.market_regime = result.get('regime', 'neutral')
                    self.regime_tp = float(result.get('take_profit', 12.0))
                    self.regime_reason = result.get('reason', '')
                    self.config.dynamic_tp = self.regime_tp
                    log.info(f"Market regime: {self.market_regime} - TP set to {self.regime_tp}%")
                    return

            # Fallback rule-based regime detection
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
        log.info(f"Fallback regime: {self.market_regime} - TP {self.regime_tp}%")

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

    def _auto_execute(self, signals):
        for signal in signals:
            action = signal.get('action')
            confidence = signal.get('confidence', 0)
            pair = signal.get('pair')
            if action not in ('buy', 'sell'):
                continue
            if confidence < 68:
                log.info(f"Auto-execute skipped {pair} - confidence {confidence}% below 65%")
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
                            log.info(f"Skipping auto-sell {pair} — open manual position (managed by TP/SL monitor)")
                            self.risk_manager.release_lock(pair)
                            continue
                    except Exception:
                        pass

                # Check if we should pyramid (add to existing position)
                if action == 'buy' and getattr(self.config, 'pyramid_enabled', False):
                    try:
                        sym = pair.replace('/', '')
                        prices = self.trader.client.get_symbol_ticker(symbol=sym)
                        current_price = float(prices['price'])
                        should_add, reason = self.trader.should_pyramid(sym, current_price)
                        if should_add:
                            # Extra safety: only pyramid if signal confidence is decent
                            if confidence < 68:
                                log.info(f"Pyramid skipped for {pair}: confidence {confidence}% too low")
                                self.risk_manager.release_lock(pair)
                                continue
                            log.info(f"Pyramid opportunity for {pair}: {reason}")
                            # Use smaller pyramid size
                            pyramid_pct = getattr(self.config, 'pyramid_size_pct', 3.0)
                            result = self.trader.execute_trade(pair, 'buy', pyramid_pct)
                            self.risk_manager.record_trade(pair)
                            log.info(f"Pyramid buy executed for {pair}: {result}")
                            self.risk_manager.release_lock(pair)
                            continue
                    except Exception as e:
                        log.debug(f"Pyramid check error for {pair}: {e}")

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
                    # Record buy time BEFORE executing so hold check works immediately
                    self.risk_manager.record_trade(pair)
                result = self.trader.execute_trade(pair, action, self.config.max_trade_pct)
                self.risk_manager.record_trade(pair)  # Start cooldown
                log.info(f"Auto-executed: {action.upper()} {pair} - confidence {confidence}% - {result}")
                # Send Telegram notification for auto trades
                try:
                    import requests as _req
                    if self.config.telegram_token and self.config.telegram_chat_id:
                        icon = '🟢' if action == 'buy' else '🔴'
                        tp = getattr(self.config, 'dynamic_tp', self.config.default_tp_pct)
                        msg = (
                            f"{icon} *AUTO {action.upper()} {pair}*\n"
                            f"Confidence: {confidence}%\n"
                            f"Market: {self.market_regime.upper()}\n"
                            f"TP: {tp}% · SL: {self.config.default_sl_pct}%"
                        )
                        _req.post(
                            f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                            json={'chat_id': str(self.config.telegram_chat_id), 'text': msg, 'parse_mode': 'Markdown'},
                            timeout=8
                        )
                except Exception as te:
                    log.debug(f"Telegram notification error: {te}")
            except Exception as e:
                log.error(f"Auto-execute failed for {pair}: {e}")
            finally:
                self.risk_manager.release_lock(pair)  # Always release lock

    def get_latest_signals(self):
        if not self.latest_signals:
            self.refresh_signals()
            # Check trailing stops every cycle
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

        # Get per-pair RSI thresholds using tier-based auto-assignment
        if hasattr(self.config, 'get_pair_rsi'):
            base_buy, base_sell = self.config.get_pair_rsi(symbol)
        else:
            base_buy, base_sell = self.config.rsi_buy, self.config.rsi_sell

        # Auto-adapt for market regime
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
                f"RSI: {round(rsi, 1)}, MACD: {macd_signal} ({round(macd_hist, 4)})\n"
                f"Price: {round(price, 4)}, 50MA: {round(ma50, 4)}, 200MA: {ma200_str}\n"
                f"Above 50MA: {price > ma50}, BB position: {round(bb_pos, 0)}%\n"
                f"Volume ratio: {round(volume_ratio, 1)}x, 24h change: {round(price_change, 2)}%\n"
                f"RSI buy threshold: {indicators.get('rsi_buy_threshold', self.config.rsi_buy)}, sell threshold: {indicators.get('rsi_sell_threshold', self.config.rsi_sell)} (auto-adjusted for {self.market_regime} regime)\n"
                f"Recommended TP for this regime: {self.regime_tp}%\n\n"
                f"Return ONLY JSON: "
                f'{{"action":"buy"|"sell"|"watch"|"hold","confidence":0-100,"reason":"2-3 sentences",'
                f'"rsi":{round(rsi)},"macd":"bullish"|"bearish"|"neutral","trend":"up"|"down"|"sideways"}}'
            )

            response = requests.post(
                ANTHROPIC_API,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )

            if response.status_code != 200:
                return self._fallback_signal(symbol, indicators)

            data = response.json()
            text = data['content'][0]['text'].strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1:
                return self._fallback_signal(symbol, indicators)

            signal_data = json.loads(text[start:end])
            signal_data['pair'] = symbol.replace('USDT', '') + '/USDT'
            return signal_data

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
        action = 'hold'
        confidence = 50
        reasons = []

        if rsi < rsi_buy_threshold:
            reasons.append("RSI oversold at " + str(round(rsi, 1)))

            # ── FEAR & GREED FILTER ──────────────────────────────────────
            fg = indicators.get('fear_greed', 50)
            if fg < 15:
                log.info(f"Fear & Greed {fg} - Extreme Fear, skipping buy {symbol}")
                action = 'hold'
                confidence = 35
            # ── TREND FILTER ─────────────────────────────────────────────
            elif ma50 > 0 and price < ma50 * 0.985:
                log.info(f"Trend filter: {symbol} downtrend, skipping buy")
                action = 'hold'
                confidence = 40
            # ── VOLUME FILTER ─────────────────────────────────────────────
            elif volume_ratio < 0.8:
                log.info(f"Volume filter: {symbol} low volume ({volume_ratio:.1f}x), skipping")
                action = 'hold'
                confidence = 45
            else:
                action = 'buy'
                confidence += 15

                # ── BOLLINGER BAND BOUNCE CONFIRMATION ──────────────────
                bb_bounce = indicators.get('bb_bounce', False)
                if bb_bounce:
                    confidence += 12
                    reasons.append("Price bouncing off lower Bollinger Band")
                elif bb and bb.get('position', 50) > 40:
                    confidence -= 8  # Not near lower band - penalise

                # ── RSI DIVERGENCE BONUS ─────────────────────────────────
                rsi_div = indicators.get('rsi_divergence', 'none')
                if rsi_div == 'bullish':
                    confidence += 15
                    reasons.append("Bullish RSI divergence detected")

                # ── STOCHASTIC RSI CONFIRMATION ──────────────────────────
                stoch = indicators.get('stoch_rsi', 50)
                if stoch < 20:
                    confidence += 10
                    reasons.append("Stochastic RSI oversold")
                elif stoch > 50:
                    confidence -= 5  # Stoch not confirming oversold

                # ── STANDARD CONFIRMATIONS ───────────────────────────────
                if price > ma50:
                    confidence += 8
                if volume_ratio > 1.5:
                    confidence += 8
                if fg > 40:  # Greed/neutral market = more confidence
                    confidence += 5
                if fg < 25:  # Fear market = less confidence
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
        """Stochastic RSI - faster momentum signal 0-100"""
        try:
            if len(closes) < rsi_period + stoch_period:
                return 50.0
            # Calculate RSI series
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
        """Detect bullish RSI divergence - price lower low but RSI higher low"""
        try:
            if len(closes) < lookback + 14:
                return 'none'
            # Get recent price lows and RSI at those points
            recent_closes = closes[-(lookback+14):]
            rsi_now = self._rsi(recent_closes)
            rsi_prev = self._rsi(recent_closes[:-5])
            price_now = closes[-1]
            price_prev = min(closes[-(lookback):-5])
            # Bullish divergence: price made lower low but RSI made higher low
            if price_now < price_prev and rsi_now > rsi_prev + 2:
                return 'bullish'
            # Bearish divergence: price made higher high but RSI made lower high
            if price_now > price_prev and rsi_now < rsi_prev - 2:
                return 'bearish'
            return 'none'
        except Exception:
            return 'none'

    def _detect_bb_bounce(self, closes, bb):
        """Detect if price is bouncing off lower Bollinger Band"""
        try:
            if not bb or 'lower' not in bb:
                return False
            lower = bb['lower']
            price = closes[-1]
            prev_price = closes[-2] if len(closes) >= 2 else price
            # Price touched lower band and is now moving up
            near_lower = price < lower * 1.015  # Within 1.5% of lower band
            bouncing = price > prev_price        # Price increasing
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
