"""
AutoTrader Pro - Main Flask Server
Added: market regime endpoint
"""

from flask import Flask, jsonify, request, send_file
from datetime import datetime
from flask_cors import CORS
import threading
import logging
import os
import json
import requests as req
from trader import Trader
from sniper import ListingSniper
from manual_positions import ManualPositionManager
from signals import SignalEngine
from risk_manager import RiskManager
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

config = Config()
trader = None
sniper = None
signal_engine = None
manual_manager = None
risk_manager = RiskManager(config)


def send_telegram(message):
    if not config.telegram_token:
        log.warning("Telegram: TELEGRAM_TOKEN not set")
        return
    if not config.telegram_chat_id:
        log.warning("Telegram: TELEGRAM_CHAT_ID not set")
        return
    try:
        log.info(f"Sending Telegram message to chat_id={config.telegram_chat_id}")
        r = req.post(
            f"https://api.telegram.org/bot{config.telegram_token}/sendMessage",
            json={'chat_id': str(config.telegram_chat_id), 'text': message, 'parse_mode': 'Markdown'},
            timeout=10
        )
        data = r.json()
        if data.get('ok'):
            log.info("Telegram message sent successfully")
        else:
            log.warning(f"Telegram failed: {data.get('description', 'Unknown error')}")
    except Exception as e:
        log.warning(f"Telegram exception: {e}")


def init_trader():
    global trader, sniper, signal_engine, manual_manager
    if not (config.api_key and config.api_secret):
        log.warning("API credentials not set - skipping trader init")
        return
    try:
        log.info("init_trader: creating Trader...")
        trader = Trader(config)

        log.info("init_trader: creating SignalEngine...")
        signal_engine = SignalEngine(config, trader, risk_manager)

        log.info("init_trader: creating ListingSniper...")
        sniper = ListingSniper(config, trader, risk_manager)

        log.info("init_trader: creating ManualPositionManager...")
        manual_manager = ManualPositionManager(config, trader)
        signal_engine.manual_manager = manual_manager

        if config.sniper_active:
            log.info("init_trader: starting sniper thread...")
            threading.Thread(target=sniper.run, daemon=True, name="sniper").start()

        log.info("init_trader: starting signal engine thread...")
        threading.Thread(target=signal_engine.run, daemon=True, name="signal_engine").start()

        log.info("init_trader: starting manual position monitor thread...")
        threading.Thread(target=manual_manager.run, daemon=True, name="manual_monitor").start()

        log.info("Trader, Sniper, Signal Engine and Manual Position Manager initialised.")
        send_telegram("✅ *AutoTrader Pro Started*\nBot is live and monitoring markets.")
    except Exception as e:
        log.error(f"Failed to initialise trader: {e}", exc_info=True)


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'connected': trader is not None,
        'sniper': sniper.active if sniper else False,
        'auto_mode': config.auto_mode,
        'version': '1.0.0'
    })


@app.route('/api/config', methods=['POST'])
def set_config():
    data = request.json
    config.api_key = data.get('api_key', '')
    config.api_secret = data.get('api_secret', '')
    config.save()
    threading.Thread(target=init_trader, daemon=True).start()
    return jsonify({'success': True})


@app.route('/api/portfolio')
def get_portfolio():
    if not trader:
        return jsonify({'error': 'Not connected'}), 400
    try:
        return jsonify(trader.get_portfolio())
    except Exception as e:
        log.error(f"Portfolio error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/history')
def portfolio_history():
    try:
        history = config.load_portfolio_history()
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'history': [], 'error': str(e)})


@app.route('/api/portfolio/refresh-deposits', methods=['POST'])
def refresh_deposits():
    """Force a fresh pull of deposit/withdrawal history from Binance.
    Use this after you've topped up your account and want the 'overall portfolio
    increase' % to recalculate immediately (otherwise it waits up to 6 hours
    for the cache to expire)."""
    if not trader:
        return jsonify({'error': 'Not connected'}), 400
    try:
        deposits, withdrawals = trader.refresh_deposit_cache()
        return jsonify({
            'success': True,
            'total_deposited_usdt': deposits,
            'total_withdrawn_usdt': withdrawals,
            'net_invested_usdt': round(deposits - withdrawals, 2)
        })
    except Exception as e:
        log.error(f"Deposit refresh error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/prices')
def get_prices():
    if not trader:
        return jsonify({'pairs': []}), 200
    try:
        pairs = trader.get_prices()
        return jsonify({'pairs': pairs if pairs else []})
    except Exception as e:
        log.error(f"Prices error: {e}")
        return jsonify({'pairs': [], 'error': str(e)}), 200


@app.route('/api/signals')
def get_signals():
    if not signal_engine:
        return jsonify({'signals': []})
    try:
        return jsonify({'signals': signal_engine.get_latest_signals()})
    except Exception as e:
        return jsonify({'signals': [], 'error': str(e)})


@app.route('/api/signals/refresh', methods=['POST'])
def refresh_signals_now():
    if not signal_engine:
        return jsonify({'error': 'Signal engine not initialised'}), 400
    try:
        log.info("Manual signal refresh requested")
        signal_engine.detect_market_regime()
        signal_engine.refresh_signals()
        return jsonify({
            'success': True,
            'signals': signal_engine.get_latest_signals(),
            'market_regime': signal_engine.market_regime,
            'count': len(signal_engine.get_latest_signals())
        })
    except Exception as e:
        log.error(f"Manual refresh failed: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/stats')
def get_stats():
    """Per-pair win rate, PnL calendar, and portfolio allocation.

    Calendar now relies on the FIFO-corrected PnL from get_real_trade_history(),
    so per-day PnL is accurate even when a sell closes a pyramided position.
    """
    try:
        trades = trader.get_real_trade_history() if trader else []
        prices = trader.get_prices() if trader else []

        # Per-pair stats
        pair_stats = {}
        for t in trades:
            pair = t.get('pair', '')
            if not pair:
                continue
            if pair not in pair_stats:
                pair_stats[pair] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': 0}
            if t.get('side') == 'sell':
                pnl = t.get('pnl', 0) or 0
                pair_stats[pair]['total_pnl'] = round(pair_stats[pair]['total_pnl'] + pnl, 2)
                pair_stats[pair]['trades'] += 1
                if pnl > 0:
                    pair_stats[pair]['wins'] += 1
                elif pnl < 0:
                    pair_stats[pair]['losses'] += 1

        pair_performance = []
        for pair, s in pair_stats.items():
            total = s['wins'] + s['losses']
            win_rate = round((s['wins'] / total * 100) if total > 0 else 0)
            pair_performance.append({
                'pair': pair,
                'win_rate': win_rate,
                'total_pnl': s['total_pnl'],
                'trades': s['trades'],
                'wins': s['wins'],
                'losses': s['losses'],
                'rating': 'good' if win_rate >= 60 and s['total_pnl'] > 0 else 'poor' if win_rate < 40 or s['total_pnl'] < -2 else 'ok'
            })
        pair_performance.sort(key=lambda x: x['total_pnl'], reverse=True)

        # PnL calendar - last 30 days, build from time_ms to avoid timezone
        # ambiguity in the stored 'date' string (use the same server-local
        # timezone the user already sees in trade history).
        from datetime import datetime, timedelta
        today = datetime.now().date()
        calendar = {}
        for i in range(30):
            d = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            calendar[d] = {'pnl': 0.0, 'trades': 0, 'wins': 0, 'losses': 0}

        for t in trades:
            if t.get('side') != 'sell':
                continue
            # Prefer time_ms for accuracy, fall back to 'date' field
            if t.get('time_ms'):
                try:
                    d = datetime.fromtimestamp(int(t['time_ms']) / 1000).strftime('%Y-%m-%d')
                except Exception:
                    d = t.get('date', '')
            else:
                d = t.get('date', '')
            if d not in calendar:
                continue
            pnl = t.get('pnl', 0) or 0
            calendar[d]['pnl'] = round(calendar[d]['pnl'] + pnl, 2)
            calendar[d]['trades'] += 1
            if pnl > 0:
                calendar[d]['wins'] += 1
            elif pnl < 0:
                calendar[d]['losses'] += 1

        calendar_list = [{'date': d, **v} for d, v in sorted(calendar.items())]

        # Portfolio allocation
        allocation = []
        total_val = sum(p.get('value_usdt', 0) for p in prices) if prices else 0
        for p in prices:
            val = p.get('value_usdt', 0)
            if val >= 1.0:
                allocation.append({
                    'symbol': p.get('symbol'),
                    'value': val,
                    'pct': round((val / total_val * 100) if total_val > 0 else 0, 1)
                })
        allocation.sort(key=lambda x: x['value'], reverse=True)

        return jsonify({
            'pair_performance': pair_performance,
            'calendar': calendar_list,
            'allocation': allocation
        })
    except Exception as e:
        log.error(f"Stats error: {e}")
        return jsonify({'pair_performance': [], 'calendar': [], 'allocation': []}), 200


@app.route('/api/regime')
def get_regime():
    if not signal_engine:
        return jsonify({'regime': 'neutral', 'take_profit': 6.0, 'reason': 'Bot not initialised'})
    try:
        return jsonify(signal_engine.get_regime())
    except Exception as e:
        return jsonify({'regime': 'neutral', 'take_profit': 6.0, 'reason': str(e)})


@app.route('/api/trade', methods=['POST'])
def execute_trade():
    if not trader:
        return jsonify({'error': 'Not connected'}), 400
    data = request.json
    pair = data.get('pair')
    action = data.get('action')
    confidence = float(data.get('confidence', 0))
    is_manual = data.get('manual', False)

    if is_manual:
        log.info(f"Manual trade requested: {action} {pair} - bypassing cooldown")
    else:
        approved, reason = risk_manager.check_trade(pair, action, confidence)
        if not approved:
            return jsonify({'error': f'Risk manager blocked: {reason}'}), 400
    try:
        result = trader.execute_trade(pair, action, config.max_trade_pct)
    except Exception as e:
        log.error(f"Trade execution error: {e}")
        risk_manager.release_lock(pair)
        send_telegram(f"⚠️ *Trade Failed* - {action.upper()} {pair}\nError: {str(e)[:100]}")
        return jsonify({'error': str(e)}), 500

    risk_manager.release_lock(pair)
    risk_manager.record_trade(pair)

    if action == 'buy' and is_manual and manual_manager:
        try:
            fills = result.get('fills', []) if isinstance(result, dict) else []
            if fills:
                total_qty = sum(float(f['qty']) for f in fills)
                entry_price = sum(float(f['price']) * float(f['qty']) for f in fills) / total_qty
            else:
                entry_price = float(result.get('price', 0)) if isinstance(result, dict) else 0
            qty = float(result.get('executedQty', 0)) if isinstance(result, dict) else 0
            usdt = entry_price * qty
            if entry_price > 0 and qty > 0:
                manual_manager.add_position(pair, entry_price, qty, usdt)
        except Exception as me:
            log.warning(f"Could not register manual position: {me}")

    try:
        regime = signal_engine.market_regime if signal_engine else 'neutral'
        source = '👤 Manual' if is_manual else f'🤖 AI Signal ({confidence}%)'
        icon = '🟢' if action == 'buy' else '🔴'
        tp = getattr(config, 'dynamic_tp', getattr(config, 'default_tp_pct', 12))
        sl = getattr(config, 'default_sl_pct', 4)
        order_id = result.get('orderId', 'N/A') if isinstance(result, dict) else 'N/A'
        send_telegram(
            f"{icon} *{action.upper()} {pair}*\n"
            f"Source: {source}\n"
            f"Market: {regime.upper()}\n"
            f"TP: {tp}% · SL: {sl}%\n"
            f"Order ID: {order_id}"
        )
    except Exception as te:
        log.warning(f"Telegram notification error (trade was successful): {te}")

    return jsonify({'success': True, 'result': result if isinstance(result, dict) else {}})


@app.route('/api/history')
def get_history():
    try:
        if trader:
            return jsonify({'trades': trader.get_real_trade_history()})
        return jsonify({'trades': config.load_trade_history()})
    except Exception as e:
        return jsonify({'trades': [], 'error': str(e)})


@app.route('/api/manual/positions')
def get_manual_positions():
    if not manual_manager:
        return jsonify({'positions': {}})
    return jsonify({'positions': manual_manager.get_all()})

@app.route('/api/manual/close', methods=['POST'])
def close_manual_position():
    pair = request.json.get('pair')
    if not pair or not manual_manager:
        return jsonify({'error': 'Invalid request'}), 400
    manual_manager.remove_position(pair)
    return jsonify({'success': True})

@app.route('/api/sniper/status')
def sniper_status():
    if not sniper:
        return jsonify({'active': False, 'detections': [], 'watching': 0})
    return jsonify({
        'active': sniper.active,
        'detections': sniper.recent_detections[-10:],
        'watching': len(sniper.seen_symbols),
        'status': 'running' if sniper.active else 'paused'
    })


@app.route('/api/sniper/positions')
def sniper_positions():
    if not trader:
        return jsonify({'error': 'Not initialised'}), 400

    positions = []
    detections_by_symbol = {}
    try:
        if sniper and hasattr(sniper, 'recent_detections'):
            for d in sniper.recent_detections:
                sym = d.get('symbol', '')
                if sym and d.get('status') in ('bought', 'tp_hit', 'sl_hit', 'time_exit'):
                    detections_by_symbol[sym] = d
    except Exception:
        pass

    try:
        account = trader.client.get_account()
        trading_pairs = set(getattr(config, 'trading_pairs', []))

        for b in account['balances']:
            asset = b['asset']
            if asset in ('USDT', 'BNB', 'USDC', 'BUSD', 'FDUSD'):
                continue
            bal = float(b['free']) + float(b['locked'])
            if bal == 0:
                continue

            symbol = asset + 'USDT'
            if symbol in trading_pairs:
                continue

            try:
                current_price = float(trader.client.get_symbol_ticker(symbol=symbol)['price'])
                value = bal * current_price
                if value < 1.0:
                    continue
            except Exception:
                continue

            entry = {
                'symbol': symbol,
                'pair': f"{asset}/USDT",
                'qty': round(bal, 6),
                'current_price': current_price,
                'value_usdt': round(value, 2),
            }

            det = detections_by_symbol.get(symbol)
            if det and det.get('buy_price'):
                entry['buy_price'] = det['buy_price']
                entry['change_pct'] = round(((current_price - det['buy_price']) / det['buy_price']) * 100, 2)
                entry['unrealised_usdt'] = round((current_price - det['buy_price']) * bal, 2)
                entry['bought_at'] = det.get('bought_at')
                entry['tp_pct'] = det.get('tp_pct', float(getattr(config, 'sniper_tp_pct', 20)))
                entry['sl_pct'] = det.get('sl_pct', float(getattr(config, 'sniper_sl_pct', 10)))
                entry['time_remaining_min'] = det.get('time_remaining_min')
                entry['monitoring'] = det.get('monitoring', False)
                entry['status'] = det.get('status', 'bought')
                entry['source'] = 'monitor'
            else:
                try:
                    trades = trader.client.get_my_trades(symbol=symbol, limit=20)
                    buys = [t for t in trades if t.get('isBuyer')]
                    if buys:
                        most_recent = max(buys, key=lambda t: int(t['time']))
                        entry['buy_price'] = float(most_recent['price'])
                        entry['change_pct'] = round(((current_price - entry['buy_price']) / entry['buy_price']) * 100, 2)
                        entry['unrealised_usdt'] = round((current_price - entry['buy_price']) * bal, 2)
                        entry['bought_at'] = datetime.fromtimestamp(int(most_recent['time']) / 1000).isoformat(timespec='seconds')
                except Exception:
                    pass
                entry['tp_pct'] = float(getattr(config, 'sniper_tp_pct', 20))
                entry['sl_pct'] = float(getattr(config, 'sniper_sl_pct', 10))
                entry['monitoring'] = False
                entry['status'] = 'orphaned'
                entry['source'] = 'inferred'

            positions.append(entry)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    positions.sort(key=lambda p: abs(p.get('unrealised_usdt', 0)), reverse=True)

    return jsonify({
        'positions': positions,
        'count': len(positions),
    })


@app.route('/api/telegram/test')
def test_telegram():
    if not config.telegram_token or not config.telegram_chat_id:
        return jsonify({
            'success': False,
            'error': 'TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in Railway Variables',
            'token_set': bool(config.telegram_token),
            'chat_id_set': bool(config.telegram_chat_id)
        })
    try:
        import requests as test_req
        msg = (
            "✅ *AutoTrader Pro - Test Message*\n\n"
            "Telegram is connected and working!\n"
            "You will receive alerts for every trade.\n\n"
            "_Sent from your Railway bot_"
        )
        r = test_req.post(
            f"https://api.telegram.org/bot{config.telegram_token}/sendMessage",
            json={'chat_id': config.telegram_chat_id, 'text': msg, 'parse_mode': 'Markdown'},
            timeout=10
        )
        data = r.json()
        if data.get('ok'):
            return jsonify({'success': True, 'message': 'Test message sent! Check Telegram.'})
        else:
            return jsonify({'success': False, 'error': data.get('description', 'Unknown error'), 'response': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sniper/test')
def test_sniper():
    if not sniper:
        return jsonify({'error': 'Sniper not initialised'}), 400
    try:
        info = trader.client.get_exchange_info()
        current = {s['symbol'] for s in info['symbols']
                  if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'}
        return jsonify({
            'sniper_active': sniper.active,
            'pairs_watching': len(sniper.seen_symbols),
            'pairs_on_binance': len(current),
            'difference': len(current - sniper.seen_symbols),
            'new_since_seed': list(current - sniper.seen_symbols)[:10],
            'message': 'Sniper is working correctly' if len(current - sniper.seen_symbols) == 0 else f'{len(current-sniper.seen_symbols)} untracked pairs detected'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sniper/simulate', methods=['POST'])
def simulate_sniper_detection():
    if not sniper:
        return jsonify({'error': 'Sniper not initialised'}), 400

    body = request.json or {}
    do_buy = body.get('buy', False)
    pair = body.get('pair') or (list(sniper.seen_symbols)[0] if sniper.seen_symbols else None)
    if not pair:
        return jsonify({'error': 'No pair available'}), 400

    from datetime import datetime
    detection = {
        'symbol': pair,
        'detected_at': datetime.now().isoformat(timespec='seconds'),
        'mode': 'SIMULATED',
        'note': 'Forced via /api/sniper/simulate'
    }

    if not hasattr(sniper, 'recent_detections'):
        sniper.recent_detections = []
    sniper.recent_detections.append(detection)

    send_telegram(
        f"🎯 *SNIPER TEST - SIMULATED DETECTION*\n"
        f"Pair: `{pair}`\n"
        f"Mode: simulation (no real buy placed)\n"
        f"This confirms detection + alert pipeline is working."
    )

    result_summary = {'detection': detection, 'real_buy_placed': False}

    if do_buy:
        try:
            r = trader.execute_trade(pair.replace('USDT', '/USDT'), 'buy', 1.0)
            result_summary['real_buy_placed'] = True
            result_summary['trade_result'] = r if isinstance(r, dict) else {'raw': str(r)}
            send_telegram(f"🟢 SNIPER TEST: real buy placed on {pair}")
        except Exception as e:
            result_summary['buy_error'] = str(e)
            log.error(f"Sniper simulate buy failed: {e}")

    return jsonify(result_summary)



def toggle_sniper():
    if not sniper:
        return jsonify({'error': 'Not initialised'}), 400
    sniper.active = request.json.get('active', False)
    config.sniper_active = sniper.active
    config.save()
    return jsonify({'active': sniper.active})


@app.route('/api/pyramid/status')
def pyramid_status():
    if not signal_engine or not trader:
        return jsonify({'error': 'Not initialised'}), 400

    try:
        from signals import PYRAMID_MIN_CONFIDENCE
    except Exception:
        PYRAMID_MIN_CONFIDENCE = 65

    out = {
        'pyramid_enabled': bool(getattr(config, 'pyramid_enabled', False)),
        'pyramid_max_adds': getattr(config, 'pyramid_max_adds', 2),
        'pyramid_drop_trigger_pct': getattr(config, 'pyramid_drop_trigger', 4.0),
        'pyramid_max_drop_pct': getattr(config, 'pyramid_max_drop', 10.0),
        'min_confidence_required': PYRAMID_MIN_CONFIDENCE,
        'positions': []
    }

    if not out['pyramid_enabled']:
        out['summary'] = 'Pyramid Mode is OFF - turn it on in Settings'
        return jsonify(out)

    latest_signals = {s.get('pair', ''): s for s in (signal_engine.get_latest_signals() or [])}

    try:
        account = trader.client.get_account()
    except Exception as e:
        return jsonify({'error': f'Could not fetch balances: {e}'}), 400

    for b in account['balances']:
        asset = b['asset']
        if asset in ('USDT', 'BNB', 'USDC', 'BUSD', 'FDUSD'):
            continue
        bal = float(b['free']) + float(b['locked'])
        if bal == 0:
            continue

        symbol = asset + 'USDT'
        pair_slash = asset + '/USDT'

        if symbol not in getattr(config, 'trading_pairs', []):
            continue

        info = {'pair': pair_slash, 'balance': bal}

        try:
            price_data = trader.client.get_symbol_ticker(symbol=symbol)
            current_price = float(price_data['price'])
            info['current_price'] = current_price
            info['value_usdt'] = round(bal * current_price, 2)
        except Exception as e:
            info['error'] = f'Price fetch failed: {e}'
            out['positions'].append(info)
            continue

        if info['value_usdt'] < 2.0:
            info['status'] = 'SKIP_DUST'
            info['reason'] = f'Position value ${info["value_usdt"]} below $2 - treated as dust'
            out['positions'].append(info)
            continue

        signal = latest_signals.get(pair_slash)
        if signal:
            info['ai_action'] = signal.get('action')
            info['ai_confidence'] = signal.get('confidence')
            if signal.get('action') == 'sell':
                info['status'] = 'BLOCKED_AI_SAYS_SELL'
                info['reason'] = 'AI wants to exit - pyramid declined'
                out['positions'].append(info)
                continue
            if (signal.get('confidence') or 0) < PYRAMID_MIN_CONFIDENCE:
                info['status'] = 'BLOCKED_LOW_CONFIDENCE'
                info['reason'] = f'AI confidence {signal.get("confidence")}% below {PYRAMID_MIN_CONFIDENCE}% required for pyramid'
                out['positions'].append(info)
                continue
        else:
            info['ai_action'] = 'no signal yet'

        try:
            should_add, reason = trader.should_pyramid(symbol, current_price)
            info['should_pyramid_returns'] = bool(should_add)
            info['trader_reason'] = str(reason)
            if should_add:
                info['status'] = 'WOULD_PYRAMID_NOW'
            else:
                info['status'] = 'NO_PYRAMID_YET'
        except AttributeError:
            info['status'] = 'ERROR'
            info['reason'] = 'trader.should_pyramid method does not exist - check trader.py'
        except Exception as e:
            info['status'] = 'ERROR'
            info['reason'] = f'should_pyramid raised: {e}'

        out['positions'].append(info)

    would = [p for p in out['positions'] if p.get('status') == 'WOULD_PYRAMID_NOW']
    errors = [p for p in out['positions'] if p.get('status') == 'ERROR']
    if errors:
        out['summary'] = f'⚠️ {len(errors)} pair(s) errored - trader.should_pyramid may be broken. See positions[].reason for details.'
    elif would:
        out['summary'] = f'✅ Pyramid READY to fire on {len(would)} position(s) next cycle: {", ".join(p["pair"] for p in would)}'
    elif out['positions']:
        out['summary'] = '✅ Pyramid backend wired up correctly - no positions meet conditions right now (waiting for price to drop %s%% from last entry)' % out['pyramid_drop_trigger_pct']
    else:
        out['summary'] = 'No real positions held - pyramid cannot fire without existing positions (it adds to positions, does not open them)'

    return jsonify(out)


@app.route('/api/risk/clear-locks', methods=['POST'])
def clear_risk_locks():
    if not risk_manager:
        return jsonify({'error': 'Risk manager not initialised'}), 400
    cleared = []
    errors = []
    try:
        pairs = list(getattr(config, 'trading_pairs', []))
        for symbol in pairs:
            pair_slash = symbol.replace('USDT', '/USDT')
            try:
                risk_manager.release_lock(pair_slash)
                cleared.append(pair_slash)
            except Exception as e:
                errors.append({'pair': pair_slash, 'error': str(e)})
        log.info(f"Manual lock clear: released {len(cleared)} pairs")
        return jsonify({
            'success': True,
            'cleared_count': len(cleared),
            'cleared_pairs': cleared,
            'errors': errors
        })
    except Exception as e:
        log.error(f"Clear locks failed: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/diagnose')
def diagnose_auto_execute():
    if not signal_engine:
        return jsonify({'error': 'Signal engine not initialised'}), 400

    try:
        from signals import AUTO_EXECUTE_MIN_CONFIDENCE
    except Exception:
        AUTO_EXECUTE_MIN_CONFIDENCE = 60

    out = {
        'auto_mode': bool(getattr(config, 'auto_mode', False)),
        'min_confidence_threshold': AUTO_EXECUTE_MIN_CONFIDENCE,
        'market_regime': getattr(signal_engine, 'market_regime', 'unknown'),
        'trading_pairs': list(getattr(config, 'trading_pairs', [])),
        'pairs': []
    }

    if not out['auto_mode']:
        out['summary'] = ('AUTO MODE IS OFF. The bot will never auto-buy or auto-sell. '
                         'Turn it on in Settings -> Trading Mode -> Full Auto Mode.')

    latest = []
    try:
        latest = signal_engine.get_latest_signals() or []
    except Exception as e:
        out['signals_error'] = str(e)

    signal_by_pair = {s.get('pair', '').replace('/USDT', 'USDT'): s for s in latest}

    for symbol in out['trading_pairs']:
        pair_slash = symbol.replace('USDT', '/USDT')
        info = {'pair': pair_slash}
        signal = signal_by_pair.get(symbol)
        if not signal:
            info['status'] = 'NO_SIGNAL'
            info['reason'] = 'No signal generated yet for this pair (signal engine may not have cycled, or signal generation failed)'
            out['pairs'].append(info)
            continue

        action = signal.get('action', 'hold')
        confidence = signal.get('confidence', 0)
        info['action'] = action
        info['confidence'] = confidence
        info['rsi'] = signal.get('rsi')
        info['signal_reason'] = signal.get('reason', '')[:120]

        if not out['auto_mode']:
            info['status'] = 'AUTO_MODE_OFF'
            info['reason'] = 'Auto Mode toggle is off - bot only suggests, never trades'
        elif action not in ('buy', 'sell'):
            info['status'] = 'SKIPPED'
            info['reason'] = f'Action is "{action}" - bot only auto-executes buy/sell'
        elif confidence < AUTO_EXECUTE_MIN_CONFIDENCE:
            info['status'] = 'SKIPPED'
            info['reason'] = f'Confidence {confidence}% below threshold {AUTO_EXECUTE_MIN_CONFIDENCE}%'
        else:
            try:
                approved, reason = risk_manager.check_trade(pair_slash, action, confidence)
                try:
                    risk_manager.release_lock(pair_slash)
                except Exception:
                    pass
                if approved:
                    info['status'] = 'WOULD_EXECUTE'
                    info['reason'] = f'Bot would auto-{action} this pair on next cycle'
                else:
                    info['status'] = 'BLOCKED'
                    info['reason'] = f'Risk manager: {reason}'
            except Exception as e:
                info['status'] = 'ERROR'
                info['reason'] = f'Risk check error: {e}'

        if action == 'sell' and trader:
            try:
                base = symbol.replace('USDT', '')
                acct = trader.client.get_account()
                bal = next((float(b['free']) + float(b['locked']) for b in acct['balances'] if b['asset'] == base), 0.0)
                price_data = trader.client.get_symbol_ticker(symbol=symbol)
                value = bal * float(price_data['price'])
                info['holdings_usdt'] = round(value, 2)
                if value < 2.0:
                    info['status'] = 'SKIPPED'
                    info['reason'] = f'Sell signal but holdings value (${value:.2f}) below $2 minimum'
            except Exception:
                pass

        out['pairs'].append(info)

    if 'summary' not in out:
        would_exec = [p for p in out['pairs'] if p.get('status') == 'WOULD_EXECUTE']
        blocked = [p for p in out['pairs'] if p.get('status') == 'BLOCKED']
        skipped_low_conf = [p for p in out['pairs'] if p.get('status') == 'SKIPPED' and 'below threshold' in p.get('reason', '')]
        if would_exec:
            out['summary'] = f"{len(would_exec)} pair(s) ready to auto-execute on next cycle: {', '.join(p['pair'] for p in would_exec)}"
        elif blocked:
            out['summary'] = f"All buy/sell signals are being BLOCKED by risk manager. Common reasons: cooldown not elapsed, or daily loss limit hit."
        elif skipped_low_conf:
            out['summary'] = f"All signals are below the {AUTO_EXECUTE_MIN_CONFIDENCE}% confidence threshold. Lower threshold in signals.py or wait for stronger setups."
        else:
            out['summary'] = "No actionable buy/sell signals right now. AI is recommending hold/watch on all pairs."

    return jsonify(out)


@app.route('/api/dust/convert', methods=['POST'])
def convert_dust():
    if not trader:
        return jsonify({'error': 'Not connected'}), 400

    try:
        account = trader.client.get_account()
        dust_assets = []

        for b in account['balances']:
            asset = b['asset']
            free = float(b['free'])
            locked = float(b['locked'])
            total = free + locked

            if total == 0 or asset in ('USDT', 'BNB', 'BUSD', 'USDC', 'FDUSD'):
                continue

            try:
                price_data = trader.client.get_symbol_ticker(symbol=asset + 'USDT')
                price = float(price_data['price'])
                value_usdt = total * price
                if 0 < value_usdt < 10:
                    dust_assets.append({'asset': asset, 'value_usdt': round(value_usdt, 4)})
            except Exception:
                pass

        if not dust_assets:
            return jsonify({
                'converted': False,
                'message': 'No dust to convert (all balances either zero, large enough, or already stablecoin)'
            })

        asset_list = [d['asset'] for d in dust_assets]
        log.info(f"Converting dust to BNB: {asset_list}")
        try:
            result = trader.client.transfer_dust(asset=','.join(asset_list))
        except Exception:
            result = trader.client.transfer_dust(asset=asset_list)

        try:
            if config.telegram_token and config.telegram_chat_id:
                assets_str = ', '.join(asset_list)
                send_telegram(f"🧹 *Dust converted to BNB*\nAssets: {assets_str}")
        except Exception:
            pass

        return jsonify({
            'converted': True,
            'assets': dust_assets,
            'asset_count': len(dust_assets),
            'message': f'Converted {len(dust_assets)} dust assets to BNB'
        })

    except Exception as e:
        log.error(f"Dust conversion failed: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/insights')
def get_insights():
    regime = 'neutral'
    regime_tp = 6.0
    try:
        regime = signal_engine.market_regime if signal_engine else 'neutral'
        regime_tp = signal_engine.regime_tp if signal_engine else 6.0
    except Exception:
        pass

    insights = []
    watchlist = []
    pair_recs = []
    risk_warning = ""

    try:
        import requests as req_lib

        def get_binance_ticker(symbol):
            r = req_lib.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}", timeout=5)
            return r.json() if r.status_code == 200 else {}

        def get_rsi(symbol, interval='1h', period=14):
            r = req_lib.get(f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={period+1}", timeout=5)
            if r.status_code != 200: return None
            closes = [float(k[4]) for k in r.json()]
            gains, losses = [], []
            for i in range(1, len(closes)):
                diff = closes[i] - closes[i-1]
                gains.append(max(diff, 0))
                losses.append(max(-diff, 0))
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            if avg_loss == 0: return 100
            rs = avg_gain / avg_loss
            return round(100 - (100 / (1 + rs)), 1)

        analysis_pairs = list(getattr(config, 'trading_pairs', ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT']))
        watch_candidates = ['AVAXUSDT','INJUSDT','SUIUSDT','APTUSDT','LINKUSDT','DOTUSDT','NEARUSDT']

        pair_data = {}
        for symbol in analysis_pairs + watch_candidates:
            try:
                t = get_binance_ticker(symbol)
                rsi = get_rsi(symbol)
                if t and 'lastPrice' in t:
                    pair_data[symbol] = {
                        'price': float(t['lastPrice']),
                        'change': float(t['priceChangePercent']),
                        'volume': float(t['quoteVolume']),
                        'high': float(t['highPrice']),
                        'low': float(t['lowPrice']),
                        'rsi': rsi
                    }
            except Exception:
                pass

        btc = pair_data.get('BTCUSDT', {})
        btc_change = btc.get('change', 0)
        btc_rsi = btc.get('rsi', 50)
        btc_price = btc.get('price', 0)

        if btc_price > 0:
            if btc_change > 3:
                insights.append({"title": f"BTC Up {btc_change:.1f}% - Momentum Strong",
                    "body": f"Bitcoin is trading at ${btc_price:,.0f}, up {btc_change:.1f}% in 24h. RSI at {btc_rsi} - {'still room to run' if btc_rsi < 65 else 'approaching overbought, watch for pullback'}. Altcoins typically follow within 4-12 hours.",
                    "type": "bullish"})
            elif btc_change < -3:
                insights.append({"title": f"BTC Down {abs(btc_change):.1f}% - Caution",
                    "body": f"Bitcoin dropped to ${btc_price:,.0f}, down {abs(btc_change):.1f}% in 24h. RSI at {btc_rsi} - {'oversold, possible bounce zone' if btc_rsi < 35 else 'still room to fall further'}. Bot trend filter is protecting against buying into this dip.",
                    "type": "bearish"})
            else:
                insights.append({"title": f"BTC Consolidating at ${btc_price:,.0f}",
                    "body": f"Bitcoin is ranging with only {btc_change:+.1f}% change in 24h. RSI at {btc_rsi} - neutral territory. Consolidation phases often precede large moves - {'bull bias given regime' if regime == 'bullish' else 'watch for direction before adding positions'}.",
                    "type": "neutral"})

        oversold = [(s, d) for s, d in pair_data.items() if d.get('rsi') and d['rsi'] < 35 and s in analysis_pairs]
        overbought = [(s, d) for s, d in pair_data.items() if d.get('rsi') and d['rsi'] > 70 and s in analysis_pairs]

        if oversold:
            names = ', '.join([s.replace('USDT','') for s,_ in oversold[:3]])
            rsis = ', '.join([str(d['rsi']) for _,d in oversold[:3]])
            insights.append({"title": f"Oversold: {names}",
                "body": f"{names} showing RSI readings of {rsis} - technically oversold. {'Trend filter active - bot will only buy if price is near 50MA.' if len(oversold) > 0 else ''} Watch for RSI reversal confirmation before entry.",
                "type": "bullish"})
        elif overbought:
            names = ', '.join([s.replace('USDT','') for s,_ in overbought[:3]])
            insights.append({"title": f"Overbought Warning: {names}",
                "body": f"{names} RSI above 70 - overbought territory. Bot will auto-generate sell signals. If holding these, take-profit orders are already active via OCO on Binance.",
                "type": "warning"})
        else:
            insights.append({"title": "RSI Neutral Across Pairs",
                "body": f"All monitored pairs showing RSI between 35-70 - no extreme readings. Market in balance. Bot confidence threshold at 72% means it will wait for stronger signals before trading.",
                "type": "neutral"})

        high_vol = [(s, d) for s, d in pair_data.items() if d.get('volume', 0) > 50_000_000 and abs(d.get('change', 0)) > 2]
        if high_vol:
            top = sorted(high_vol, key=lambda x: x[1]['volume'], reverse=True)[0]
            sym, data = top
            name = sym.replace('USDT', '')
            insights.append({"title": f"High Volume Alert: {name}",
                "body": f"{name} showing ${data['volume']/1e6:.0f}M in 24h volume with {data['change']:+.1f}% price move. High volume moves are more likely to sustain direction. {'Bot has this pair active.' if sym in analysis_pairs else 'Not currently in bot - consider adding via Railway TRADING_PAIRS.'}",
                "type": "bullish" if data['change'] > 0 else "bearish"})

        wl_candidates = [(s, d) for s, d in pair_data.items() if s in watch_candidates and d.get('rsi')]
        wl_candidates.sort(key=lambda x: x[1]['change'], reverse=True)
        for sym, data in wl_candidates[:3]:
            name = sym.replace('USDT', '')
            rsi = data['rsi']
            change = data['change']
            if rsi < 40:
                signal = "buy"
                reason = f"RSI oversold at {rsi} with {change:+.1f}% 24h move - potential bounce candidate."
            elif rsi > 65:
                signal = "avoid"
                reason = f"RSI at {rsi} - overbought. Wait for pullback before entry."
            else:
                signal = "watch"
                reason = f"RSI at {rsi}, {change:+.1f}% in 24h - neutral setup, monitor for breakout."
            watchlist.append({"symbol": sym, "name": name, "reason": reason, "signal": signal})

        for sym in ['AVAXUSDT', 'INJUSDT', 'SUIUSDT', 'NEARUSDT', 'APTUSDT']:
            if sym in pair_data:
                d = pair_data[sym]
                name = sym.replace('USDT', '')
                in_bot = sym in analysis_pairs
                if d.get('rsi') and d['rsi'] < 40 and d['volume'] > 20_000_000:
                    pair_recs.append({"symbol": sym, "name": name, "action": "add" if not in_bot else "keep",
                        "reason": f"RSI {d['rsi']} oversold + ${d['volume']/1e6:.0f}M volume. Strong entry setup right now."})
                elif d.get('rsi') and d['rsi'] > 70:
                    pair_recs.append({"symbol": sym, "name": name, "action": "keep" if in_bot else "avoid",
                        "reason": f"RSI {d['rsi']} overbought - wait for pullback before adding."})
                else:
                    pair_recs.append({"symbol": sym, "name": name, "action": "keep" if in_bot else "add",
                        "reason": f"{d['change']:+.1f}% 24h, RSI {d['rsi']} - solid mid-cap with good liquidity on Binance."})
            if len(pair_recs) >= 3:
                break

        losers = [(s, d) for s, d in pair_data.items() if s in analysis_pairs and d.get('change', 0) < -5]
        if losers:
            names = ', '.join([s.replace('USDT','') for s,_ in losers])
            risk_warning = f"{names} down over 5% today - check your OCO stop loss orders are active in Binance app."
        elif regime == 'bearish':
            risk_warning = "Bear market regime detected - bot is using tighter RSI thresholds and lower take-profit targets. Reduce position sizes if uncertain."
        elif btc_rsi and btc_rsi > 75:
            risk_warning = f"BTC RSI at {btc_rsi} - historically high. Consider reducing exposure or tightening stop losses on open positions."
        else:
            risk_warning = "No major risk signals detected. Ensure OCO orders are active in Binance for all open positions."

    except Exception as e:
        log.warning(f"Insights data fetch error: {e}")
        insights = [{"title": "Market data temporarily unavailable", "body": "Could not fetch live Binance data. Check Railway logs for details.", "type": "warning"}]
        watchlist = []
        pair_recs = []
        risk_warning = "Check Railway logs - insights data fetch failed."

    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    ai_powered = False
    if gemini_key and pair_data:
        try:
            data_summary = f"Live Binance data {datetime.now().strftime('%Y-%m-%d %H:%M')}:\n"
            for sym, d in list(pair_data.items())[:12]:
                rsi_str = f" RSI:{d['rsi']}" if d.get('rsi') else ""
                data_summary += f"{sym}: ${d['price']:.4f} {d['change']:+.1f}%{rsi_str}\n"
            data_summary += f"Regime: {regime}. TP: {regime_tp}%. Bot pairs: {','.join(analysis_pairs)}"

            ai_prompt = (
                "You are a professional crypto trader analysing live market data. Be specific and direct.\n\n"
                + data_summary +
                "\n\nUsing this real data, give specific actionable analysis. Reference actual prices and RSI values. "
                "Suggest specific coins based on current conditions. Do NOT be generic.\n"
                "Return ONLY valid JSON, no markdown, no explanation:\n"
                '{"insights":['
                '{"title":"under 8 words","body":"2 specific sentences with real numbers","type":"bullish|bearish|neutral|warning"},'
                '{"title":"under 8 words","body":"2 specific sentences with real numbers","type":"bullish|bearish|neutral|warning"},'
                '{"title":"under 8 words","body":"2 specific sentences with real numbers","type":"bullish|bearish|neutral|warning"}'
                '],'
                '"watchlist":['
                '{"symbol":"XYZUSDT","name":"CoinName","reason":"specific reason with price/RSI data","signal":"buy|watch|avoid"},'
                '{"symbol":"XYZUSDT","name":"CoinName","reason":"specific reason with price/RSI data","signal":"buy|watch|avoid"},'
                '{"symbol":"XYZUSDT","name":"CoinName","reason":"specific reason with price/RSI data","signal":"buy|watch|avoid"}'
                '],'
                '"pair_recommendations":['
                '{"symbol":"XYZUSDT","name":"CoinName","action":"add|remove|keep","reason":"specific current reason"},'
                '{"symbol":"XYZUSDT","name":"CoinName","action":"add|remove|keep","reason":"specific current reason"},'
                '{"symbol":"XYZUSDT","name":"CoinName","action":"add|remove|keep","reason":"specific current reason"}'
                '],'
                '"risk_warning":"1 specific sentence with actual data"}'
            )

            r = req_lib.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": ai_prompt}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1200}
                },
                timeout=30
            )

            if r.status_code == 200:
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                text = text.replace("```json", "").replace("```", "").strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    ai_result = json.loads(text[start:end])
                    insights = ai_result.get("insights", insights)
                    watchlist = ai_result.get("watchlist", watchlist)
                    pair_recs = ai_result.get("pair_recommendations", pair_recs)
                    risk_warning = ai_result.get("risk_warning", risk_warning)
                    ai_powered = True
                    log.info("Gemini AI insights generated successfully")
                else:
                    log.warning(f"Could not parse Gemini response: {text[:200]}")
            else:
                log.warning(f"Gemini API error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            log.warning(f"Gemini insights failed: {e}")

    return jsonify({
        "regime": regime,
        "updated": datetime.now().strftime("%H:%M"),
        "ai_powered": ai_powered,
        "insights": insights[:3],
        "watchlist": watchlist[:3],
        "pair_recommendations": pair_recs[:3],
        "risk_warning": risk_warning
    })


# =============================================================================
# Extra settings persistence
# =============================================================================
import os as _os_extra
_EXTRA_SETTINGS_PATH = '/data/extra_settings.json'
_EXTRA_KEYS = [
    'trailing_stop_enabled',
    'trailing_stop_pct',
    'trailing_stop_activate_pct',
    'trailing_breakeven_trigger',
    'concentration_alert_pct',
    'approval_mode',
]


def _load_extra_settings():
    try:
        with open(_EXTRA_SETTINGS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_extra_settings(data):
    try:
        _os_extra.makedirs(_os_extra.path.dirname(_EXTRA_SETTINGS_PATH), exist_ok=True)
        with open(_EXTRA_SETTINGS_PATH, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        log.warning(f"Could not save extra settings: {e}")


def _apply_extras_to_config():
    extras = _load_extra_settings()
    for k, v in extras.items():
        try:
            setattr(config, k, v)
        except Exception:
            pass
    return extras


@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        payload = request.json or {}
        config.update(payload)
        config.save()

        existing = _load_extra_settings()
        for k in _EXTRA_KEYS:
            if k in payload:
                v = payload[k]
                if k in ('trailing_stop_enabled', 'approval_mode'):
                    v = bool(v) if not isinstance(v, str) else (v.lower() == 'true')
                else:
                    try:
                        v = float(v)
                    except Exception:
                        pass
                existing[k] = v
                try:
                    setattr(config, k, v)
                except Exception:
                    pass
        _save_extra_settings(existing)
        log.info(f"Settings saved - extras: { {k: existing.get(k) for k in _EXTRA_KEYS if k in existing} }")

        if risk_manager:
            risk_manager.reload(config)
        return jsonify({'success': True})

    result = config.to_dict()
    extras = _load_extra_settings()
    for k in _EXTRA_KEYS:
        if k in extras:
            result[k] = extras[k]
        elif hasattr(config, k):
            result[k] = getattr(config, k)
    return jsonify(result)


try:
    _apply_extras_to_config()
    log.info("Extra settings (trailing stop / concentration) loaded from disk")
except Exception as _e:
    log.debug(f"No extra settings to load: {_e}")


# =============================================================================
# STARTUP
# =============================================================================
log.info("=" * 60)
log.info("AutoTrader Pro - module loaded, scheduling background init")
log.info("=" * 60)

threading.Thread(target=init_trader, daemon=True, name="init_trader").start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    log.info(f"AutoTrader Pro starting Flask on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
