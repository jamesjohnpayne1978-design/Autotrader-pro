"""
AutoTrader Pro - Binance Trading Engine
Fixed: P&L calculation, trade cooldown, quantity precision
"""

import logging
import math
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException

log = logging.getLogger(__name__)


class Trader:
    def __init__(self, config):
        self.config = config
        self.client = Client(config.api_key, config.api_secret)
        self._verify_connection()
        self._symbol_info_cache = {}
        self._last_trade_time = {}

    def _verify_connection(self):
        try:
            self.client.ping()
            log.info("Binance connection established.")
        except BinanceAPIException as e:
            log.error(f"Binance connection failed: {e}")
            raise

    def is_on_cooldown(self, pair):
        cooldown = self.config.trade_cooldown_minutes
        if pair in self._last_trade_time:
            elapsed = (datetime.now() - self._last_trade_time[pair]).total_seconds() / 60
            if elapsed < cooldown:
                log.info(f"Cooldown active for {pair}: {cooldown - elapsed:.1f} mins remaining")
                return True
        return False

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
        today_pnl = sum(t.get('pnl', 0) for t in today_trades)

        # Save portfolio snapshot for chart
        self._save_portfolio_snapshot(round(total_usdt, 2))

        return {
            'total_usdt': round(total_usdt, 2),
            'positions': positions,
            'trades_today': len(today_trades),
            'open_positions': len([p for p in positions if p['asset'] != 'USDT']),
            'win_rate': win_rate,
            'pnl_today': round(today_pnl, 2),
            'pnl_pct': 0.0
        }

    def _save_portfolio_snapshot(self, value):
        try:
            snapshots = self.config.load_portfolio_history()
            now = datetime.now()
            # Only save one snapshot per hour
            if snapshots:
                last = snapshots[-1]
                last_time = datetime.fromisoformat(last['time'])
                if (now - last_time).total_seconds() < 3600:
                    return
            snapshots.append({
                'time': now.isoformat(),
                'value': value,
                'date': now.strftime('%Y-%m-%d %H:%M')
            })
            # Keep last 90 days of hourly snapshots
            snapshots = snapshots[-2160:]
            self.config.save_portfolio_history(snapshots)
        except Exception as e:
            log.debug(f"Snapshot save failed: {e}")

    def get_portfolio_history(self):
        return self.config.load_portfolio_history()

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
        # Check cooldown
        if self.is_on_cooldown(pair):
            raise ValueError(f"Trade cooldown active for {pair}")

        symbol = pair.replace('/', '')
        base = symbol.replace('USDT', '')

        if action == 'buy':
            usdt_balance = self._get_balance('USDT')
            amount_usdt = usdt_balance * (pct_of_portfolio / 100)
            amount_usdt = max(15, min(amount_usdt, usdt_balance * 0.95))
            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            raw_quantity = amount_usdt / price
            quantity = self._adjust_quantity(symbol, raw_quantity)
            if quantity <= 0:
                raise ValueError(f"Calculated quantity is zero for {symbol}")
            order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            buy_price = float(order.get('fills', [{}])[0].get('price', price)) if order.get('fills') else price
            log.info(f"BUY {symbol}: qty={quantity} at ${buy_price:.4f}")
            self._last_trade_time[pair] = datetime.now()
            self._log_trade(pair, 'buy', order, buy_price, quantity)

        elif action == 'sell':
            quantity = self._get_balance(base)
            if quantity <= 0:
                raise ValueError(f"No {base} balance to sell")
            quantity = self._adjust_quantity(symbol, quantity * 0.999)
            if quantity <= 0:
                raise ValueError(f"Adjusted sell quantity is zero for {symbol}")
            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            sell_price = float(order.get('fills', [{}])[0].get('price', price)) if order.get('fills') else price
            log.info(f"SELL {symbol}: qty={quantity} at ${sell_price:.4f}")
            self._last_trade_time[pair] = datetime.now()
            pnl = self._calculate_pnl(pair, sell_price, quantity)
            self._log_trade(pair, 'sell', order, sell_price, quantity, pnl)
        else:
            raise ValueError(f"Unknown action: {action}")

        return {'orderId': order['orderId'], 'status': order['status']}

    def _calculate_pnl(self, pair, sell_price, quantity):
        try:
            history = self.config.load_trade_history()
            # Find most recent buy for this pair
            for trade in history:
                if trade.get('pair') == pair and trade.get('side') == 'buy':
                    buy_price = trade.get('price', 0)
                    if buy_price > 0:
                        pnl = (sell_price - buy_price) * quantity
                        return round(pnl, 2)
        except Exception as e:
            log.debug(f"PnL calculation error: {e}")
        return 0.0

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
                return 0
            min_notional_filter = next(
                (f for f in info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')),
                None
            )
            if min_notional_filter:
                min_notional = float(min_notional_filter.get('minNotional', 0))
                price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
                if quantity * price < min_notional:
                    return 0
            return quantity
        except Exception as e:
            log.error(f"Quantity adjustment error: {e}")
            return round(quantity, 5)

    def _log_trade(self, pair, action, order, price=0, quantity=0, pnl=0):
        trade = {
            'pair': pair,
            'side': action,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'orderId': order['orderId'],
            'status': order['status'],
            'price': round(price, 6),
            'quantity': round(quantity, 6),
            'pnl': pnl,
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
            return {'success': False, 'error': str(e)}
