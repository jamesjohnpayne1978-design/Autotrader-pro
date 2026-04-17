"""
AutoTrader Pro - Binance Trading Engine
Fixed: Real PnL from Binance trade history, OCO orders, gain%, free USDT
"""

import logging
import math
from datetime import datetime, timedelta
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
        self._open_oco_orders = {}

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
        free_usdt = 0.0
        for b in balances:
            asset = b['asset']
            amount = float(b['free']) + float(b['locked'])
            if asset == 'USDT':
                total_usdt += amount
                free_usdt = float(b['free'])
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

        self._save_portfolio_snapshot(round(total_usdt, 2))

        # Real daily PnL from portfolio snapshots
        pnl_today = 0.0
        pnl_pct = 0.0
        try:
            snapshots = self.config.load_portfolio_history()
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_snaps = [s for s in snapshots if s.get('date', '').startswith(today_str)]
            if today_snaps and today_snaps[0]['value'] > 0:
                start_val = today_snaps[0]['value']
                pnl_today = round(total_usdt - start_val, 2)
                pnl_pct = round((pnl_today / start_val) * 100, 2)
        except Exception:
            pass

        # Trade counts
        history = self.get_real_trade_history()
        today = datetime.now().strftime('%Y-%m-%d')
        trades_today = len([t for t in history if t.get('date', '') == today])
        closed = [t for t in history if t.get('side') == 'sell' and t.get('pnl', 0) != 0]
        wins = [t for t in closed if t.get('pnl', 0) > 0]
        win_rate = round(len(wins) / len(closed) * 100) if closed else None

        return {
            'total_usdt': round(total_usdt, 2),
            'positions': positions,
            'trades_today': trades_today,
            'open_positions': len([p for p in positions if p['asset'] != 'USDT' and p.get('value_usdt', 0) >= 1.0]),
            'win_rate': win_rate,
            'pnl_today': pnl_today,
            'pnl_pct': pnl_pct,
            'free_usdt': round(free_usdt, 2)
        }

    def _save_portfolio_snapshot(self, value):
        try:
            snapshots = self.config.load_portfolio_history()
            now = datetime.now()
            if snapshots:
                last_time = datetime.fromisoformat(snapshots[-1]['time'])
                if (now - last_time).total_seconds() < 3600:
                    return
            snapshots.append({
                'time': now.isoformat(),
                'value': value,
                'date': now.strftime('%Y-%m-%d %H:%M')
            })
            self.config.save_portfolio_history(snapshots[-2160:])
        except Exception as e:
            log.debug(f"Snapshot save failed: {e}")

    def get_portfolio_history(self):
        return self.config.load_portfolio_history()

    def get_prices(self):
        results = []
        history = self.get_real_trade_history()
        tp_pct = getattr(self.config, 'dynamic_tp', self.config.default_tp_pct)
        sl_pct = self.config.default_sl_pct

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
                pair_name = f"{base}/USDT"

                # Find last buy price from real history
                buy_price = None
                for trade in history:
                    if trade.get('pair') == pair_name and trade.get('side') == 'buy':
                        buy_price = trade.get('price', 0)
                        if buy_price and buy_price > 0:
                            break

                gain_pct = None
                to_tp = None
                if buy_price and buy_price > 0 and holding * price >= 1.0:
                    gain_pct = round(((price - buy_price) / buy_price) * 100, 2)
                    to_tp = round(tp_pct - gain_pct, 2)

                results.append({
                    'symbol': pair_name,
                    'base': base,
                    'price': round(price, 6),
                    'change': round(float(stats['priceChangePercent']), 2),
                    'volume': float(stats['volume']),
                    'holdings': round(holding, 6),
                    'value_usdt': round(holding * price, 2),
                    'buy_price': round(buy_price, 6) if buy_price else None,
                    'gain_pct': gain_pct,
                    'to_tp': to_tp,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct
                })
            except Exception as e:
                log.warning(f"Could not fetch {symbol}: {e}")
        return results

    def get_real_trade_history(self):
        """
        Pull real trade history from Binance for all configured pairs.
        Matches buys to sells and calculates real PnL per trade.
        """
        try:
            all_trades = []
            for symbol in self.config.trading_pairs:
                try:
                    trades = self.client.get_my_trades(symbol=symbol, limit=50)
                    for t in trades:
                        all_trades.append({
                            'symbol': symbol,
                            'pair': symbol.replace('USDT', '') + '/USDT',
                            'side': 'buy' if t['isBuyer'] else 'sell',
                            'price': float(t['price']),
                            'quantity': float(t['qty']),
                            'usdt_value': float(t['quoteQty']),
                            'commission': float(t['commission']),
                            'time_ms': int(t['time']),
                            'time': datetime.fromtimestamp(int(t['time']) / 1000).strftime('%Y-%m-%d %H:%M'),
                            'date': datetime.fromtimestamp(int(t['time']) / 1000).strftime('%Y-%m-%d'),
                            'orderId': str(t['orderId']),
                        })
                except Exception as e:
                    log.debug(f"Could not fetch trades for {symbol}: {e}")

            # Sort by time
            all_trades.sort(key=lambda x: x['time_ms'], reverse=True)

            # Calculate PnL by matching sells to most recent buys per pair
            buy_prices = {}
            result = []
            # Process in chronological order for matching
            for trade in sorted(all_trades, key=lambda x: x['time_ms']):
                symbol = trade['symbol']
                if trade['side'] == 'buy':
                    buy_prices[symbol] = trade['price']
                    trade['pnl'] = 0.0
                    trade['trigger'] = 'AI Signal'
                elif trade['side'] == 'sell':
                    if symbol in buy_prices and buy_prices[symbol] > 0:
                        pnl = (trade['price'] - buy_prices[symbol]) * trade['quantity']
                        trade['pnl'] = round(pnl, 2)
                    else:
                        trade['pnl'] = 0.0
                    trade['trigger'] = 'AI Signal'

            # Return newest first
            for trade in sorted(all_trades, key=lambda x: x['time_ms'], reverse=True):
                result.append(trade)

            return result[:100]

        except Exception as e:
            log.error(f"Real trade history error: {e}")
            return self.config.load_trade_history()

    def get_klines(self, symbol, interval='1h', limit=100):
        raw = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        return [{
            'time': k[0], 'open': float(k[1]), 'high': float(k[2]),
            'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])
        } for k in raw]

    def execute_trade(self, pair, action, pct_of_portfolio):
        if self.is_on_cooldown(pair):
            raise ValueError(f"Trade cooldown active for {pair}")

        symbol = pair.replace('/', '')
        base = symbol.replace('USDT', '')

        if action == 'buy':
            usdt_balance = self._get_balance('USDT')
            amount_usdt = usdt_balance * (pct_of_portfolio / 100)
            amount_usdt = max(15, min(amount_usdt, usdt_balance * 0.95))
            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            quantity = self._adjust_quantity(symbol, amount_usdt / price)
            if quantity <= 0:
                raise ValueError(f"Calculated quantity is zero for {symbol}")
            order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
            buy_price = float(order.get('fills', [{}])[0].get('price', price)) if order.get('fills') else price
            log.info(f"BUY {symbol}: qty={quantity} at ${buy_price:.6f}")
            self._last_trade_time[pair] = datetime.now()
            self._log_trade(pair, 'buy', order, buy_price, quantity)
            self._place_oco_order(symbol, pair, quantity, buy_price)

        elif action == 'sell':
            self._cancel_oco_orders(symbol, pair)
            quantity = self._get_balance(base)
            if quantity <= 0:
                raise ValueError(f"No {base} balance to sell")
            quantity = self._adjust_quantity(symbol, quantity * 0.999)
            if quantity <= 0:
                raise ValueError(f"Adjusted sell quantity is zero for {symbol}")
            price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            sell_price = float(order.get('fills', [{}])[0].get('price', price)) if order.get('fills') else price
            log.info(f"SELL {symbol}: qty={quantity} at ${sell_price:.6f}")
            self._last_trade_time[pair] = datetime.now()
            pnl = self._calculate_pnl(pair, sell_price, quantity)
            self._log_trade(pair, 'sell', order, sell_price, quantity, pnl)
        else:
            raise ValueError(f"Unknown action: {action}")

        return {'orderId': order['orderId'], 'status': order['status']}

    def _place_oco_order(self, symbol, pair, quantity, buy_price):
        try:
            tp_pct = getattr(self.config, 'dynamic_tp', self.config.default_tp_pct)
            sl_pct = self.config.default_sl_pct
            tp_price = self._round_price(symbol, buy_price * (1 + tp_pct / 100))
            sl_price = self._round_price(symbol, buy_price * (1 - sl_pct / 100))
            sl_limit_price = self._round_price(symbol, sl_price * 0.995)
            log.info(f"Placing OCO for {symbol}: TP=${tp_price} SL=${sl_price} qty={quantity}")
            oco = self.client.create_oco_order(
                symbol=symbol, side='SELL', quantity=quantity,
                price=str(tp_price), stopPrice=str(sl_price),
                stopLimitPrice=str(sl_limit_price), stopLimitTimeInForce='GTC'
            )
            self._open_oco_orders[pair] = {
                'orderListId': oco.get('orderListId'),
                'symbol': symbol, 'tp_price': tp_price, 'sl_price': sl_price
            }
            log.info(f"OCO placed for {symbol} — TP: ${tp_price} ({tp_pct}%) | SL: ${sl_price} ({sl_pct}%)")
        except BinanceAPIException as e:
            log.warning(f"OCO order failed for {symbol}: {e}")
        except Exception as e:
            log.warning(f"OCO setup error for {symbol}: {e}")

    def _cancel_oco_orders(self, symbol, pair):
        if pair in self._open_oco_orders:
            try:
                order_info = self._open_oco_orders[pair]
                self.client.cancel_order_list(symbol=symbol, orderListId=order_info['orderListId'])
                del self._open_oco_orders[pair]
                log.info(f"Cancelled OCO orders for {symbol}")
            except Exception as e:
                log.warning(f"Could not cancel OCO for {symbol}: {e}")

    def _round_price(self, symbol, price):
        try:
            info = self._get_symbol_info(symbol)
            price_filter = next(f for f in info['filters'] if f['filterType'] == 'PRICE_FILTER')
            tick_size = float(price_filter['tickSize'])
            if tick_size > 0:
                precision = int(round(-math.log10(tick_size)))
                price = math.floor(price / tick_size) * tick_size
                return round(price, precision)
        except Exception:
            pass
        return round(price, 4)

    def _calculate_pnl(self, pair, sell_price, quantity):
        try:
            history = self.config.load_trade_history()
            for trade in history:
                if trade.get('pair') == pair and trade.get('side') == 'buy':
                    buy_price = trade.get('price', 0)
                    if buy_price > 0:
                        return round((sell_price - buy_price) * quantity, 2)
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
                (f for f in info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
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
            'pair': pair, 'side': action,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'orderId': order['orderId'], 'status': order['status'],
            'price': round(price, 6), 'quantity': round(quantity, 6),
            'pnl': pnl, 'trigger': 'AI Signal'
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
