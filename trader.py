"""
AutoTrader Pro - Binance Trading Engine
Fixed: quantity precision and minimum order handling
"""

import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import math

log = logging.getLogger(__name__)


class Trader:
    def __init__(self, config):
        self.config = config
        self.client = Client(config.api_key, config.api_secret)
        self._verify_connection()
        self._symbol_info_cache = {}

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
            amount_usdt = max(15, min(amount_usdt, usdt_balance * 0.95))

            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            raw_quantity = amount_usdt / price
            quantity = self._adjust_quantity(symbol, raw_quantity)

            log.info(f"BUY {symbol}: usdt={amount_usdt:.2f}, price={price}, qty={quantity}")

            if quantity <= 0:
                raise ValueError(f"Calculated quantity is zero for {symbol}")

            order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            log.info(f"BUY order placed: {order['orderId']}")

        elif action == 'sell':
            quantity = self._get_balance(base)
            if quantity <= 0:
                raise ValueError(f"No {base} balance to sell")
            quantity = self._adjust_quantity(symbol, quantity * 0.999)
            if quantity <= 0:
                raise ValueError(f"Adjusted quantity is zero for {symbol}")
            order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            log.info(f"SELL order placed: {order['orderId']}")
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

    def _get_symbol_info(self, symbol):
        if symbol not in self._symbol_info_cache:
            self._symbol_info_cache[symbol] = self.client.get_symbol_info(symbol)
        return self._symbol_info_cache[symbol]

    def _adjust_quantity(self, symbol, quantity):
        try:
            info = self._get_symbol_info(symbol)
            lot_filter = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')
            step_size = float(lot_filter['stepSize'])
            min_qty = float(lot_filter['minQty'])

            if step_size > 0:
                precision = int(round(-math.log10(step_size)))
                quantity = math.floor(quantity / step_size) * step_size
                quantity = round(quantity, precision)

            if quantity < min_qty:
                log.warning(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
                return 0

            # Check minimum notional value
            min_notional_filter = next(
                (f for f in info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')),
                None
            )
            if min_notional_filter:
                min_notional = float(min_notional_filter.get('minNotional', 0))
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                notional = quantity * price
                if notional < min_notional:
                    log.warning(f"Notional {notional:.2f} below minimum {min_notional} for {symbol}")
                    return 0

            return quantity
        except Exception as e:
            log.error(f"Quantity adjustment error for {symbol}: {e}")
            return round(quantity, 5)

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
            if quantity <= 0:
                return {'success': False, 'error': 'Invalid quantity'}
            order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            return {'success': True, 'orderId': order['orderId']}
        except Exception as e:
            log.error(f"Snipe failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
