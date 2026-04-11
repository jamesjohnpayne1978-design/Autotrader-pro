"""
AutoTrader Pro - AI Signal Engine
Uses Claude AI + technical indicators to generate trade signals
Auto-executes trades when auto_mode is enabled
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
        self.running = False

    def run(self):
        self.running = True
        log.info("Signal engine started.")
        while self.running:
            try:
                self.refresh_signals()
            except Exception as e:
                log.error(f"Signal refresh error: {e}")
            time.sleep(300)

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
        log.info(f"Signals refreshed: {len(signals)} signals generated")

        # Auto execute if enabled
        if self.config.auto_mode and self.risk_manager:
            self._auto_execute(signals)

    def _auto_execute(self, signals):
        """Automatically execute trades when auto_mode is on"""
        for signal in signals:
            action = signal.get('action')
            confidence = signal.get('confidence', 0)
            pair = signal.get('pair')

            if action not in ('buy', 'sell'):
                continue
            if confidence < 70:
                log.info(f"Auto-execute skipped {pair} — confidence {confidence}% below 70%")
                continue

            approved, reason = self.risk_manager.check_trade(pair, action, confidence)
            if not approved:
                log.info(f"Auto-execute blocked {pair}: {reason}")
                continue

            try:
                result = self.trader.execute_trade(pair, action, self.config.max_trade_pct)
                log.info(f"Auto-executed: {action.upper()} {pair} — confidence {confidence}% — order {result}")
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

        indicators = {
            'rsi': self._rsi(closes),
            'macd': self._macd(closes),
            'ma50': self._sma(closes, 50),
            'ma200': self._sma(closes, 200) if len(closes) >= 200 else None,
            'bb': self._bollinger(closes),
            'volume_ratio': volumes[-1] / np.mean(volumes[-20:]) if volumes else 1.0,
            'price': closes[-1],
            'price_change_24h': ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 else 0
        }

        signal = self._ai_analyse(symbol, indicators)
        return signal

    def _ai_analyse(self, symbol, indicators):
        try:
            prompt = f"""You are an expert crypto trading analyst. Analyse these technical indicators for {symbol} and generate a trading signal.

Indicators:
- RSI (14): {indicators['rsi']:.1f}
- MACD: {'Bullish crossover' if indicators['macd']['signal'] == 'bullish' else 'Bearish' if indicators['macd']['signal'] == 'bearish' else 'Neutral'}
- MACD histogram: {indicators['macd']['histogram']:.4f}
- Price: ${indicators['price']:.4f}
- 50MA: ${indicators['ma50']:.4f}
- 200MA: ${indicators['ma200']:.4f if indicators['ma200'] else 'N/A'}
- Price above 50MA: {indicators['price'] > indicators['ma50']}
- Bollinger: price at {indicators['bb']['position']:.0f}% of band
- Volume vs 20-day avg: {indicators['volume_ratio']:.1f}x
- 24h change: {indicators['price_change_24h']:.2f}%

Settings:
- RSI buy below: {self.config.rsi_buy}
- RSI sell above: {self.config.rsi_sell}
- MA cross strategy: {self.config.ma_cross_enabled}
- MACD signals: {self.config.macd_enabled}

Return ONLY a JSON object with these exact fields:
{{
  "action": "buy" | "sell" | "watch" | "hold",
  "confidence": 0-100,
  "reason": "2-3 sentence explanation of the signal",
  "rsi": {indicators['rsi']:.0f},
  "macd": "bullish" | "bearish" | "neutral",
  "trend": "up" | "down" | "sideways"
}}

Only generate buy/sell if confidence is above 60. Otherwise use watch or hold."""

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
                log.warning(f"Claude API error: {response.status_code}")
                return self._fallback_signal(symbol, indicators)

            data = response.json()
            text = data['content'][0]['text'].strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1:
                return self._fallback_signal(symbol, indicators)

            signal_data = json.loads(text[start:end])
            signal_data['pair'] = f"{symbol.replace('USDT', '')}/USDT"
            return signal_data

        except Exception as e:
            log.warning(f"AI analysis failed for {symbol}: {e}")
            return self._fallback_signal(symbol, indicators)

    def _fallback_signal(self, symbol, indicators):
        rsi = indicators['rsi']
        macd = indicators['macd']['signal']
        price = indicators['price']
        ma50 = indicators['ma50']

        action = 'hold'
        confidence = 50
        reasons = []

        if rsi < self.config.rsi_buy:
            reasons.append(f"RSI oversold at {rsi:.1f}")
            action = 'buy'
            confidence += 15
        elif rsi > self.config.rsi_sell:
            reasons.append(f"RSI overbought at {rsi:.1f}")
            action = 'sell'
            confidence += 15

        if macd == 'bullish' and action != 'sell':
            reasons.append("MACD bullish crossover")
            action = 'buy'
            confidence += 10
        elif macd == 'bearish' and action != 'buy':
            reasons.append("MACD bearish crossover")
            action = 'sell'
            confidence += 10

        if price > ma50 and action == 'buy':
            reasons.append("Price above 50MA")
            confidence += 10
        elif price < ma50 and action == 'sell':
            reasons.append("Price below 50MA")
            confidence += 10

        if not reasons:
            action = 'watch'

        return {
            'pair': f"{symbol.replace('USDT', '')}/USDT",
            'action': action,
            'confidence': min(confidence, 95),
            'reason': '. '.join(reasons) + '.' if reasons else 'No strong signal detected. Monitoring market conditions.',
            'rsi': round(rsi),
            'macd': macd,
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
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    def _sma(self, closes, period):
        if len(closes) < period:
            return closes[-1]
        return float(np.mean(closes[-period:]))

    def _ema(self, closes, period):
        if len(closes) < period:
            return closes[-1]
        k = 2 / (period + 1)
        ema = closes[0]
        for price in closes[1:]:
            ema = price * k + ema * (1 - k)
        return ema

    def _macd(self, closes):
        if len(closes) < 26:
            return {'signal': 'neutral', 'histogram': 0, 'macd': 0, 'signal_line': 0}
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(closes[-9:], 9)
        histogram = macd_line - signal_line
        signal = 'bullish' if macd_line > signal_line else 'bearish' if macd_line < signal_line else 'neutral'
        return {'signal': signal, 'histogram': histogram, 'macd': macd_line, 'signal_line': signal_line}

    def _bollinger(self, closes, period=20):
        if len(closes) < period:
            return {'upper': closes[-1], 'lower': closes[-1], 'mid': closes[-1], 'position': 50}
        recent = closes[-period:]
        mid = np.mean(recent)
        std = np.std(recent)
        upper = mid + 2 * std
        lower = mid - 2 * std
        price = closes[-1]
        band_range = upper - lower
        position = ((price - lower) / band_range * 100) if band_range > 0 else 50
        return {'upper': upper, 'lower': lower, 'mid': mid, 'position': position}
