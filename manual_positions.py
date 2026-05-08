"""
AutoTrader Pro - Manual Position Manager
Manual trades are tracked separately from the signal engine.
They exit via fixed TP/SL targets, never via RSI/MACD signals.
"""

import json
import time
import logging
import os
from datetime import datetime
from threading import Thread

log = logging.getLogger(__name__)
MANUAL_FILE = '/data/manual_positions.json'


class ManualPositionManager:
    def __init__(self, config, trader):
        self.config = config
        self.trader = trader
        self.positions = {}
        self._load()
        log.info(f"Manual position manager ready — {len(self.positions)} open positions")

    def _load(self):
        try:
            if os.path.exists(MANUAL_FILE):
                with open(MANUAL_FILE) as f:
                    self.positions = json.load(f)
                log.info(f"Loaded {len(self.positions)} manual positions from disk")
        except Exception as e:
            log.warning(f"Could not load manual positions: {e}")
            self.positions = {}

    def _save(self):
        try:
            os.makedirs('/data', exist_ok=True)
            with open(MANUAL_FILE, 'w') as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save manual positions: {e}")

    def add_position(self, pair, entry_price, quantity, usdt_value):
        """Register a new manual buy position"""
        tp_pct = getattr(self.config, 'manual_tp_pct', 10.0)
        sl_pct = getattr(self.config, 'manual_sl_pct', 5.0)
        self.positions[pair] = {
            'pair': pair,
            'entry_price': entry_price,
            'quantity': quantity,
            'usdt_value': usdt_value,
            'tp_price': round(entry_price * (1 + tp_pct / 100), 8),
            'sl_price': round(entry_price * (1 - sl_pct / 100), 8),
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'opened_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'open'
        }
        self._save()
        log.info(f"Manual position opened: {pair} @ ${entry_price:.4f} | TP: ${self.positions[pair]['tp_price']:.4f} | SL: ${self.positions[pair]['sl_price']:.4f}")

    def remove_position(self, pair):
        """Remove a manual position (after exit)"""
        if pair in self.positions:
            del self.positions[pair]
            self._save()

    def has_position(self, pair):
        """Check if we have an open manual position for this pair"""
        return pair in self.positions and self.positions[pair].get('status') == 'open'

    def get_position(self, pair):
        return self.positions.get(pair)

    def get_all(self):
        return dict(self.positions)

    def run(self):
        """Monitor manual positions and exit at TP/SL"""
        log.info("Manual position monitor started")
        while True:
            try:
                self._check_positions()
            except Exception as e:
                log.error(f"Manual position monitor error: {e}")
            time.sleep(30)  # Check every 30 seconds

    def _check_positions(self):
        if not self.positions:
            return

        for pair, pos in list(self.positions.items()):
            if pos.get('status') != 'open':
                continue
            try:
                symbol = pair.replace('/', '')
                price = float(self.trader.client.get_symbol_ticker(symbol=symbol)['price'])
                entry = pos['entry_price']
                tp = pos['tp_price']
                sl = pos['sl_price']
                gain_pct = ((price - entry) / entry) * 100

                if price >= tp:
                    log.info(f"Manual TP hit: {pair} @ ${price:.4f} (+{gain_pct:.2f}%)")
                    self._exit_position(pair, pos, price, 'TP hit')
                elif price <= sl:
                    log.info(f"Manual SL hit: {pair} @ ${price:.4f} ({gain_pct:.2f}%)")
                    self._exit_position(pair, pos, price, 'SL hit')
                else:
                    log.debug(f"Manual {pair}: ${price:.4f} | {gain_pct:+.2f}% | TP: ${tp:.4f} | SL: ${sl:.4f}")

            except Exception as e:
                log.warning(f"Could not check manual position {pair}: {e}")

    def _exit_position(self, pair, pos, exit_price, reason):
        try:
            result = self.trader.execute_trade(pair, 'sell', 100)
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
            log.info(f"Manual position exited: {pair} | {reason} | PnL: ${pnl:.2f}")

            # Send Telegram notification
            try:
                import requests as req
                if self.config.telegram_token and self.config.telegram_chat_id:
                    icon = 'TAKE PROFIT' if 'TP' in reason else 'STOP LOSS'
                    emoji = '' if 'TP' in reason else ''
                    msg = (
                        f"{emoji} *MANUAL {icon} — {pair}*\n"
                        f"Entry: ${pos['entry_price']:.4f}\n"
                        f"Exit: ${exit_price:.4f}\n"
                        f"PnL: {'+'if pnl>=0 else ''}${pnl:.2f}\n"
                        f"Held: {pos.get('opened_at', 'unknown')}"
                    )
                    req.post(
                        f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                        json={'chat_id': str(self.config.telegram_chat_id), 'text': msg, 'parse_mode': 'Markdown'},
                        timeout=8
                    )
            except Exception:
                pass

            self.remove_position(pair)
        except Exception as e:
            log.error(f"Could not exit manual position {pair}: {e}")
