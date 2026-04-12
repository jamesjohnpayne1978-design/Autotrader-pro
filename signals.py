"""
AutoTrader Pro - AI Signal Engine
Fixed: AI analysis format error + lowered threshold to 65%
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
                log.info(f"Auto-execute skipped {pair} - confidence {confidence}% below 65%")
                continue
            approved, reason = self.risk_manager.check_trade(pair, action, confidence)
            if not approved:
                log.info(f"Auto-execute blocked {pair}: {reason}")
                continue
            try:
                result = self.trader.execute_trade(pair, action, self.config.max_trade_pct)
                log.info(f"Auto-executed: {action.upper()} {pair} - confidence {confidence}% - order {result}")
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

        indicators = {
            'rsi': rsi,
            'macd': macd,
            'ma50': ma50,
            'ma200': ma200,
            'bb': bb,
            'volume_ratio': volume_ratio,
            'price': price,
            'price_change_24h': price_change
        }
        signal = self._ai_analyse(symbol, indicators)
        return signal

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
                f"You are an expert crypto trading analyst. Analyse these indicators for {symbol}:\n"
                f"RSI: {round(rsi, 1)}\n"
                f"MACD: {macd_signal}, histogram: {round(macd_hist, 4)}\n"
                f"Price: {round(price, 4)}\n"
                f"50MA: {round(ma50, 4)}\n"
                f"200MA: {ma200_str}\n"
                f"Price above 50MA: {price > ma50}\n"
                f"Bollinger position: {round(bb_pos, 0)}%\n"
                f"Volume ratio: {round(volume_ratio, 1)}x\n"
                f"24h change: {round(price_change, 2)}%\n"
                f"RSI buy below: {self.config.rsi_buy}\n"
                f"RSI sell above: {self.config.rsi_sell}\n\n"
                f"Return ONLY a JSON object:\n"
                f'{{"action":"buy"|"sell"|"watch"|"hold","confidence":0-100,"reason":"2-3 sentences","rsi":{round(rsi)},"macd":"bullish"|"bearish"|"neutral","trend":"up"|"down"|"sideways"}}\n'
                f"Only buy/sell if confidence above 60."
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
                log.warning(f"Claude API error: {response.status_code}")
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

        action = 'hold'
        confidence = 50
        reasons = []

        if rsi < self.config.rsi_buy:
            reasons.append("RSI oversold at " + str(round(rsi, 1)))
            action = 'buy'
            confidence += 15
        elif rsi > self.config.rsi_sell:
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
            reasons.append("Price above 50MA")
            confidence += 10
        elif price < ma50 and action == 'sell':
            reasons.append("Price below 50MA")
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
        for p in closes[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def _macd(self, closes):
        if len(closes) < 26:
            return {'signal': 'neutral', 'histogram': 0}
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(closes[-9:], 9)
        histogram = macd_line - signal_line
        if macd_line > signal_line:
            signal = 'bullish'
        elif macd_line < signal_line:
            signal = 'bearish'
        else:
            signal = 'neutral'
        return {'signal': signal, 'histogram': histogram}

    def _bollinger(self, closes, period=20):
        if len(closes) < period:
            return {'position': 50}
        recent = closes[-period:]
        mid = np.mean(recent)
        std = np.std(recent)
        upper = mid + 2 * std
        lower = mid - 2 * std
        price = closes[-1]
        band_range = upper - lower
        position = ((price - lower) / band_range * 100) if band_range > 0 else 50
        return {'upper': upper, 'lower': lower, 'mid': mid, 'position': position}
