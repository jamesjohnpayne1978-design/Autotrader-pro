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
    def __init__(self, config, trader, risk_manager=None):
        self.config = config
        self.trader = trader
        self.risk_manager = risk_manager
        self.latest_signals = []
        self.market_regime = 'neutral'
        self.regime_reason = 'Analysing market conditions...'
        self.regime_tp = 6.0
        self.running = False

    def run(self):
        self.running = True
        log.info("Signal engine started.")
        while self.running:
            try:
                self.detect_market_regime()
                self.refresh_signals()
            except Exception as e:
                log.error(f"Signal refresh error: {e}")
            time.sleep(300)

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
                f'{{"regime":"bullish"|"bearish"|"neutral","take_profit":4|6|8,"reason":"one sentence explanation"}}'
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
                    self.regime_tp = float(result.get('take_profit', 6.0))
                    self.regime_reason = result.get('reason', '')
                    self.config.dynamic_tp = self.regime_tp
                    log.info(f"Market regime: {self.market_regime} — TP set to {self.regime_tp}%")
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
            self.regime_tp = 8.0
            self.regime_reason = f"RSI {round(rsi)} above 55, price up {round(price_change_7d, 1)}% in 7 days and above 7MA."
        elif rsi < 45 and price_change_7d < -3 and price < ma7:
            self.market_regime = 'bearish'
            self.regime_tp = 4.0
            self.regime_reason = f"RSI {round(rsi)} below 45, price down {round(abs(price_change_7d), 1)}% in 7 days and below 7MA."
        else:
            self.market_regime = 'neutral'
            self.regime_tp = 6.0
            self.regime_reason = f"Mixed signals — RSI {round(rsi)}, 7-day change {round(price_change_7d, 1)}%."

        self.config.dynamic_tp = self.regime_tp
        log.info(f"Fallback regime: {self.market_regime} — TP {self.regime_tp}%")

    def _fallback_regime_simple(self):
        self.market_regime = 'neutral'
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
        log.info(f"Signals refreshed: {len(signals)} signals — regime: {self.market_regime} — TP: {self.regime_tp}%")
        if self.config.auto_mode and self.risk_manager:
            self._auto_execute(signals)

    def _auto_execute(self, signals):
        for signal in signals:
            action = signal.get('action')
            confidence = signal.get('confidence', 0)
            pair = signal.get('pair')
            if action not in ('buy', 'sell'):
                continue
            if confidence < 65:
                log.info(f"Auto-execute skipped {pair} — confidence {confidence}% below 65%")
                continue
            approved, reason = self.risk_manager.check_trade(pair, action, confidence)
            if not approved:
                log.info(f"Auto-execute blocked {pair}: {reason}")
                continue
            try:
                result = self.trader.execute_trade(pair, action, self.config.max_trade_pct)
                self.risk_manager.record_trade(pair)  # Start cooldown
                log.info(f"Auto-executed: {action.upper()} {pair} — confidence {confidence}% — {result}")
            except Exception as e:
                log.error(f"Auto-execute failed for {pair}: {e}")

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
            'bb': bb, 'volume_ratio': volume_ratio, 'price': price,
            'price_change_24h': price_change,
            'rsi_buy_threshold': rsi_buy_threshold,
            'rsi_sell_threshold': rsi_sell_threshold
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
            action = 'buy'
            confidence += 15
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
