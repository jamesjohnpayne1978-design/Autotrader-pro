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

        # Daily PnL from actual closed Binance trades today
        pnl_today = 0.0
        pnl_pct = 0.0
        pnl_total = 0.0
        pnl_total_pct = 0.0
        try:
            real_trades = self.get_real_trade_history()
            today_str = datetime.now().strftime('%Y-%m-%d')
            # Sum PnL from all closed (sell) trades today
            today_sells = [t for t in real_trades if t.get('side') == 'sell' and t.get('date', '') == today_str and t.get('pnl', 0) != 0]
            pnl_today = round(sum(t.get('pnl', 0) for t in today_sells), 2)
            if total_usdt > 0 and pnl_today != 0:
                pnl_pct = round((pnl_today / (total_usdt - pnl_today)) * 100, 2)
            # Total PnL from all closed trades ever
            all_sells = [t for t in real_trades if t.get('side') == 'sell' and t.get('pnl', 0) != 0]
            pnl_total = round(sum(t.get('pnl', 0) for t in all_sells), 2)
            # Total % vs first snapshot
            snapshots = self.config.load_portfolio_history()
            if snapshots and snapshots[0]['value'] > 0:
                first_val = snapshots[0]['value']
                pnl_total_pct = round(((total_usdt - first_val) / first_val) * 100, 2)
        except Exception as e:
            log.debug(f"PnL calc error: {e}")

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
            'pnl_total': pnl_total,
            'pnl_total_pct': pnl_total_pct,
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
        # Use saved history (fast) not real-time Binance history (slow/times out)
        history = self.config.load_trade_history()
        tp_pct = getattr(self.config, 'dynamic_tp', None) or self.config.default_tp_pct
        sl_pct = self.config.default_sl_pct

        for symbol in self.config.trading_pairs:
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                stats = self.client.get_ticker(symbol=symbol)
                base = symbol.replace('USDT', '')
                # Get account balance once per symbol safely
                try:
                    account = self.client.get_account()
                    holding = next(
                        (float(b['free']) + float(b['locked'])
                         for b in account['balances'] if b['asset'] == base), 0.0
                    )
                except Exception:
                    holding = 0.0

                price = float(ticker['price'])
                pair_name = f"{base}/USDT"

                # Find most recent buy price from saved history
                buy_price = None
                try:
                    # History is newest first — find most recent buy
                    # But only if there's no more recent sell (position is open)
                    found_sell = False
                    for trade in history:
                        if trade.get('pair') != pair_name:
                            continue
                        if trade.get('side') == 'sell':
                            found_sell = True
                            break  # Most recent trade is a sell — no open position
                        if trade.get('side') == 'buy' and not found_sell:
                            bp = trade.get('price', 0)
                            if bp and float(bp) > 0:
                                buy_price = float(bp)
                                break
                except Exception:
                    buy_price = None

                gain_pct = None
                to_tp = None
                try:
                    if buy_price and buy_price > 0 and holding * price >= 1.0:
                        gain_pct = round(((price - buy_price) / buy_price) * 100, 2)
                        to_tp = round(tp_pct - gain_pct, 2)
                except Exception:
                    pass

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
                # Add basic entry so pair still shows on dashboard
                try:
                    base = symbol.replace('USDT', '')
                    results.append({
                        'symbol': f"{base}/USDT",
                        'base': base,
                        'price': 0,
                        'change': 0,
                        'volume': 0,
                        'holdings': 0,
                        'value_usdt': 0,
                        'buy_price': None,
                        'gain_pct': None,
                        'to_tp': None,
                        'tp_pct': tp_pct,
                        'sl_pct': sl_pct
                    })
                except Exception:
                    pass
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

            # Consolidate partial fills with same orderId into single trades
            consolidated = {}
            for t in all_trades:
                key = t['orderId']
                if key not in consolidated:
                    consolidated[key] = t.copy()
                else:
                    # Merge fills: add quantity and value, average price
                    existing = consolidated[key]
                    total_qty = existing['quantity'] + t['quantity']
                    total_val = existing['usdt_value'] + t['usdt_value']
                    existing['price'] = round(total_val / total_qty, 6) if total_qty > 0 else existing['price']
                    existing['quantity'] = round(total_qty, 6)
                    existing['usdt_value'] = round(total_val, 2)

            all_trades = list(consolidated.values())
            all_trades.sort(key=lambda x: x['time_ms'], reverse=True)

            # Calculate PnL by matching sells to most recent buys per pair
            buy_prices = {}
            result = []
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
            # Calculate weighted average fill price across all fills
            fills = order.get('fills', [])
            if fills:
                total_qty = sum(float(f['qty']) for f in fills)
                buy_price = sum(float(f['price']) * float(f['qty']) for f in fills) / total_qty if total_qty > 0 else price
            else:
                buy_price = price
            log.info(f"Fill price for {symbol}: ${buy_price:.6f} ({len(fills)} fills)")
            log.info(f"BUY {symbol}: qty={quantity} at ${buy_price:.6f}")
            self._last_trade_time[pair] = datetime.now()
            usdt_spent = buy_price * quantity
            self._log_trade(pair, 'buy', order, buy_price, quantity, usdt_value=usdt_spent)
            # Place OCO order — wrap in try so a failed OCO doesn't lose the trade
            try:
                self._place_oco_order(symbol, pair, quantity, buy_price)
            except Exception as oco_err:
                log.warning(f"OCO order failed for {symbol} (trade still executed): {oco_err}")
            # Start trailing stop tracking
            if getattr(self.config, 'trailing_stop_enabled', False):
                try:
                    self.init_trailing_stop(symbol, buy_price)
                except Exception:
                    pass

        elif action == 'sell':
            # Cancel any open OCO orders first (releases locked balance)
            self._cancel_all_open_orders(symbol)
            import time; time.sleep(1)  # Wait for cancellation to process
            # Use free + locked balance since OCO is now cancelled
            quantity = self._get_total_balance(base)
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
            self.clear_trailing_stop(symbol)
        else:
            raise ValueError(f"Unknown action: {action}")

        return {'orderId': order['orderId'], 'status': order['status']}

    def _place_oco_order(self, symbol, pair, quantity, buy_price):
        try:
            tp_pct = getattr(self.config, 'dynamic_tp', None) or self.config.default_tp_pct
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

    # ─── TRAILING STOP ──────────────────────────────────────────────
    _trailing_stops = {}  # {symbol: {'buy_price': x, 'highest': x, 'trail_pct': x}}

    def init_trailing_stop(self, symbol, buy_price):
        """Start tracking a trailing stop for a position"""
        trail_pct = getattr(self.config, 'trailing_stop_pct', 2.0)
        breakeven_trigger = getattr(self.config, 'trailing_breakeven_trigger', 3.0)
        self.__class__._trailing_stops[symbol] = {
            'buy_price': buy_price,
            'highest': buy_price,
            'trail_pct': trail_pct,
            'breakeven_trigger': breakeven_trigger,
            'stop_price': buy_price * (1 - self.config.default_sl_pct / 100)
        }
        log.info(f"Trailing stop init for {symbol} @ ${buy_price:.4f}")

    def update_trailing_stop(self, symbol, current_price):
        """Update trailing stop — returns (should_sell, reason) if stop hit"""
        state = self.__class__._trailing_stops.get(symbol)
        if not state:
            return False, None

        buy_price = state['buy_price']
        highest = state['highest']
        trail_pct = state['trail_pct']
        breakeven_trigger = state['breakeven_trigger']
        gain_pct = ((current_price - buy_price) / buy_price) * 100

        # Update highest price seen
        if current_price > highest:
            state['highest'] = current_price
            # Trail the stop up behind it
            new_stop = current_price * (1 - trail_pct / 100)
            # Never move stop down
            if new_stop > state['stop_price']:
                state['stop_price'] = new_stop
                log.info(f"Trailing stop {symbol}: moved to ${new_stop:.4f} ({trail_pct}% below ${current_price:.4f})")

        # Move to breakeven once up enough
        if gain_pct >= breakeven_trigger and state['stop_price'] < buy_price:
            state['stop_price'] = buy_price * 1.001  # Tiny above buy = breakeven
            log.info(f"Trailing stop {symbol}: moved to breakeven @ ${state['stop_price']:.4f}")

        # Check if stop hit
        if current_price <= state['stop_price']:
            gain = ((current_price - buy_price) / buy_price) * 100
            return True, f"Trailing stop hit @ ${current_price:.4f} ({gain:+.1f}% from entry)"

        return False, None

    def clear_trailing_stop(self, symbol):
        self.__class__._trailing_stops.pop(symbol, None)

    def check_all_trailing_stops(self):
        """Called every cycle — check if any trailing stops need executing"""
        if not self.__class__._trailing_stops:
            return
        for symbol in list(self.__class__._trailing_stops.keys()):
            try:
                price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
                should_sell, reason = self.update_trailing_stop(symbol, price)
                if should_sell:
                    pair = symbol.replace('USDT', '/USDT')
                    log.info(f"Trailing stop triggered for {pair}: {reason}")
                    self.execute_trade(pair, 'sell', 100)
                    self.clear_trailing_stop(symbol)
            except Exception as e:
                log.debug(f"Trailing stop check error {symbol}: {e}")

    # ─── PYRAMIDING ────────────────────────────────────────────────
    _pyramid_state = {}  # class-level: {symbol: {count, first_price, last_price}}

    def get_pyramid_state(self, symbol):
        return self.__class__._pyramid_state.get(symbol, {'count': 0, 'first_price': 0.0, 'last_price': 0.0})

    def record_pyramid_buy(self, symbol, price):
        state = self.get_pyramid_state(symbol)
        if state['count'] == 0:
            state['first_price'] = price
        state['last_price'] = price
        state['count'] = state['count'] + 1
        self.__class__._pyramid_state[symbol] = state
        log.info(f"Pyramid {symbol}: buy #{state['count']} @ ${price:.4f} (first: ${state['first_price']:.4f})")

    def reset_pyramid_state(self, symbol):
        self.__class__._pyramid_state.pop(symbol, None)
        log.info(f"Pyramid state reset for {symbol}")

    def should_pyramid(self, symbol, current_price):
        """Returns (bool, reason) — whether to add to existing position"""
        if not getattr(self.config, 'pyramid_enabled', False):
            return False, "Pyramiding disabled"
        state = self.get_pyramid_state(symbol)
        if state['count'] == 0:
            return False, "No initial position to pyramid"
        max_adds = getattr(self.config, 'pyramid_max_adds', 2)
        if state['count'] > max_adds:
            return False, f"Max pyramid adds ({max_adds}) reached"
        first_price = state['first_price']
        last_price = state['last_price']
        if first_price <= 0:
            return False, "No valid first buy price"
        drop_from_first = ((first_price - current_price) / first_price) * 100
        drop_from_last  = ((last_price  - current_price) / last_price)  * 100
        max_drop = getattr(self.config, 'pyramid_max_drop', 10.0)
        trigger  = getattr(self.config, 'pyramid_drop_trigger', 4.0)
        if drop_from_first > max_drop:
            return False, f"Down {drop_from_first:.1f}% from first buy — too risky"
        if drop_from_last < trigger:
            return False, f"Only down {drop_from_last:.1f}% from last buy (need {trigger}%)"
        return True, f"Down {drop_from_last:.1f}% from last buy — adding position #{state['count']+1}"

    def _cancel_all_open_orders(self, symbol):
        """Cancel ALL open orders for a symbol — releases locked balance"""
        try:
            open_orders = self.client.get_open_orders(symbol=symbol)
            if not open_orders:
                log.info(f"No open orders to cancel for {symbol}")
                return
            # Cancel each order individually (python-binance compatible)
            for order in open_orders:
                try:
                    self.client.cancel_order(symbol=symbol, orderId=order['orderId'])
                    log.info(f"Cancelled order {order['orderId']} for {symbol}")
                except Exception as e:
                    log.warning(f"Could not cancel order {order['orderId']}: {e}")
            log.info(f"Cancelled {len(open_orders)} orders for {symbol}")
        except Exception as e:
            log.warning(f"Could not cancel orders for {symbol}: {e}")

    def _get_total_balance(self, asset):
        """Get free + locked balance (use after cancelling orders)"""
        account = self.client.get_account()
        for b in account['balances']:
            if b['asset'] == asset:
                return float(b['free']) + float(b['locked'])
        return 0.0

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

    def _log_trade(self, pair, action, order, price=0, quantity=0, pnl=0, usdt_value=None):
        trade = {
            'pair': pair, 'side': action,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'orderId': order['orderId'], 'status': order['status'],
            'price': round(price, 6), 'quantity': round(quantity, 6),
            'usdt_value': round(usdt_value if usdt_value else price * quantity, 2),
            'pnl': pnl, 'trigger': 'AI Signal'
        }
        # Track pyramid state
        if action == 'buy':
            self.record_pyramid_buy(symbol, price)
        elif action == 'sell':
            self.reset_pyramid_state(symbol)

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
