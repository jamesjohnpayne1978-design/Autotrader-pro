"""
AutoTrader Pro - Binance Trading Engine
"""

import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException

log = logging.getLogger(__name__)


class Trader:
    def __init__(self, config):
        self.config = config
        self.client = Client(config.api_key, config.api_secret)
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.client.ping()
            log.info("Binance connection established.")
        except BinanceAPIException as e:
            log.error(f"Binance connection failed: {e}")
            raise

    def get_portfolio(self):
        account = self.client.get_account()
        balances = [b for b in account['balances']
                    if float(b['free']) + float(b['locked']) > 0]

        total_usdt = 0.0
        positions = []

        for b in balances:
            asset = b['asset']
            amount = float(b['free']) + float(b['locked'])
            if asset == 'USDT':
                total_usdt += amount
                positions.append({'asset': asset, 'amount': amount, 'value_usdt': amount})
                continue
            try:
                ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                price = float(ticker['price'])
                value = amount * price
                total_usdt += value
                positions.append({
                    'asset': asset,
                    'amount': round(amount, 6),
                    'price': round(price, 4),
                    'value_usdt': round(value, 2)
                })
            except Exception:
                pass

        history = self.config.load_trade_history()
        today = datetime.now().strftime('%Y-%m-%d')
        today_trades = [t for t in history if t.get('date') == today]
        wins = [t for t in history if t.get('pnl', 0) > 0]
        win_rate = round(len(wins) / len(history) * 100) if history else None

        return {
            'total_usdt': round(total_usdt, 2),
            'positions': positions,
            'trades_today': len(today_trades),
            'open_positions': len([p for p in positions if p['asset'] != 'USDT']),
            'win_rate': win_rate,
            'pnl_today': round(sum(t.get('pnl', 0) for t in today_trades), 2),
            'pnl_pct': 0.0
        }

    def get_prices(self):
        results = []
        for symbol in self.config.trading_pairs:
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                stats = self.client.get_ticker(symbol=symbol)
                base = symbol.replace('USDT', '')
                account = self.client.get_account()
                holding = next(
                    (float(b['free']) + float(b['locked'])
                     for b in account['balances'] if b['asset'] == base), 0.0
                )
                price = float(ticker['price'])
                results.append({
                    'symbol': f"{base}/USDT",
                    'base': base,
                    'price': round(price, 4),
                    'change': round(float(stats['priceChangePercent']), 2),
                    'volume': float(stats['volume']),
                    'holdings': round(holding, 6),
                    'value_usdt': round(holding * price, 2)
                })
            except Exception as e:
                log.warning(f"Could not fetch {symbol}: {e}")
        return results

    def get_klines(self, symbol, interval='1h', limit=100):
        raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        return [{
            'time': k[0],
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5])
        } for k in raw]

    def execute_trade(self, pair, action, pct_of_portfolio):
        symbol = pair.replace('/', '')
        base = symbol.replace('USDT', '')

        if action == 'buy':
            usdt_balance = self._get_balance('USDT')
            amount_usdt = usdt_balance * (pct_of_portfolio / 100)
            amount_usdt = max(10, min(amount_usdt, usdt_balance * 0.95))
            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            quantity = self._adjust_quantity(symbol, amount_usdt / price)
            order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            log.info(f"BUY {symbol}: qty={quantity}")

        elif action == 'sell':
            quantity = self._get_balance(base)
            if quantity <= 0:
                raise ValueError(f"No {base} balance to sell")
            quantity = self._adjust_quantity(symbol, quantity * 0.999)
            order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            log.info(f"SELL {symbol}: qty={quantity}")
        else:
            raise ValueError(f"Unknown action: {action}")

        self._log_trade(pair, action, order)
        return {'orderId': order['orderId'], 'status': order['status']}

    def _get_balance(self, asset):
        account = self.client.get_account()
        for b in account['balances']:
            if b['asset'] == asset:
                return float(b['free'])
        return 0.0

    def _adjust_quantity(self, symbol, quantity):
        info = self.client.get_symbol_info(symbol)
        step = float(next(f['stepSize'] for f in info['filters'] if f['filterType'] == 'LOT_SIZE'))
        precision = len(str(step).rstrip('0').split('.')[-1]) if '.' in str(step) else 0
        return round(quantity - (quantity % step), precision)

    def _log_trade(self, pair, action, order):
        trade = {
            'pair': pair,
            'side': action,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'orderId': order['orderId'],
            'status': order['status'],
            'pnl': 0.0,
            'trigger': 'AI Signal'
        }
        history = self.config.load_trade_history()
        history.insert(0, trade)
        self.config.save_trade_history(history[:500])

    def snipe_listing(self, symbol, usdt_amount):
        try:
            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            quantity = self._adjust_quantity(symbol, usdt_amount / price)
            order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            return {'success': True, 'orderId': order['orderId']}
        except Exception as e:
            return {'success': False, 'error': str(e)}
