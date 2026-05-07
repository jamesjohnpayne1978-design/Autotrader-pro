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
import requests as req
from trader import Trader
from sniper import ListingSniper
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
risk_manager = RiskManager(config)


def send_telegram(message):
    if not config.telegram_token or not config.telegram_chat_id:
        return
    try:
        req.post(
            f"https://api.telegram.org/bot{config.telegram_token}/sendMessage",
            json={'chat_id': config.telegram_chat_id, 'text': message, 'parse_mode': 'Markdown'},
            timeout=5
        )
    except Exception as e:
        log.warning(f"Telegram alert failed: {e}")


def init_trader():
    global trader, sniper, signal_engine
    if config.api_key and config.api_secret:
        try:
            trader = Trader(config)
            signal_engine = SignalEngine(config, trader, risk_manager)
            sniper = ListingSniper(config, trader, risk_manager)
            if config.sniper_active:
                sniper_thread = threading.Thread(target=sniper.run, daemon=True)
                sniper_thread.start()
            signal_thread = threading.Thread(target=signal_engine.run, daemon=True)
            signal_thread.start()
            log.info("Trader, Sniper and Signal Engine initialised.")
            send_telegram("✅ *AutoTrader Pro Started*\nBot is live and monitoring markets.")
        except Exception as e:
            log.error(f"Failed to initialise trader: {e}")


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
    init_trader()
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


@app.route('/api/prices')
def get_prices():
    if not trader:
        return jsonify({'pairs': []}), 200
    try:
        pairs = trader.get_prices()
        return jsonify({'pairs': pairs if pairs else []})
    except Exception as e:
        log.error(f"Prices error: {e}")
        # Return empty list instead of 500 so dashboard shows something
        return jsonify({'pairs': [], 'error': str(e)}), 200


@app.route('/api/signals')
def get_signals():
    if not signal_engine:
        return jsonify({'signals': []})
    try:
        return jsonify({'signals': signal_engine.get_latest_signals()})
    except Exception as e:
        return jsonify({'signals': [], 'error': str(e)})



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

        # Pull live Binance market data for all major pairs
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

        # Analyse current trading pairs + additional watchlist candidates
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

        # BTC analysis
        btc = pair_data.get('BTCUSDT', {})
        btc_change = btc.get('change', 0)
        btc_rsi = btc.get('rsi', 50)
        btc_price = btc.get('price', 0)

        if btc_price > 0:
            if btc_change > 3:
                insights.append({"title": f"BTC Up {btc_change:.1f}% — Momentum Strong",
                    "body": f"Bitcoin is trading at ${btc_price:,.0f}, up {btc_change:.1f}% in 24h. RSI at {btc_rsi} — {'still room to run' if btc_rsi < 65 else 'approaching overbought, watch for pullback'}. Altcoins typically follow within 4-12 hours.",
                    "type": "bullish"})
            elif btc_change < -3:
                insights.append({"title": f"BTC Down {abs(btc_change):.1f}% — Caution",
                    "body": f"Bitcoin dropped to ${btc_price:,.0f}, down {abs(btc_change):.1f}% in 24h. RSI at {btc_rsi} — {'oversold, possible bounce zone' if btc_rsi < 35 else 'still room to fall further'}. Bot trend filter is protecting against buying into this dip.",
                    "type": "bearish"})
            else:
                insights.append({"title": f"BTC Consolidating at ${btc_price:,.0f}",
                    "body": f"Bitcoin is ranging with only {btc_change:+.1f}% change in 24h. RSI at {btc_rsi} — neutral territory. Consolidation phases often precede large moves — {'bull bias given regime' if regime == 'bullish' else 'watch for direction before adding positions'}.",
                    "type": "neutral"})

        # Find oversold opportunities
        oversold = [(s, d) for s, d in pair_data.items() if d.get('rsi') and d['rsi'] < 35 and s in analysis_pairs]
        overbought = [(s, d) for s, d in pair_data.items() if d.get('rsi') and d['rsi'] > 70 and s in analysis_pairs]

        if oversold:
            names = ', '.join([s.replace('USDT','') for s,_ in oversold[:3]])
            rsis = ', '.join([str(d['rsi']) for _,d in oversold[:3]])
            insights.append({"title": f"Oversold: {names}",
                "body": f"{names} showing RSI readings of {rsis} — technically oversold. {'Trend filter active — bot will only buy if price is near 50MA.' if len(oversold) > 0 else ''} Watch for RSI reversal confirmation before entry.",
                "type": "bullish"})
        elif overbought:
            names = ', '.join([s.replace('USDT','') for s,_ in overbought[:3]])
            insights.append({"title": f"Overbought Warning: {names}",
                "body": f"{names} RSI above 70 — overbought territory. Bot will auto-generate sell signals. If holding these, take-profit orders are already active via OCO on Binance.",
                "type": "warning"})
        else:
            insights.append({"title": "RSI Neutral Across Pairs",
                "body": f"All monitored pairs showing RSI between 35-70 — no extreme readings. Market in balance. Bot confidence threshold at 72% means it will wait for stronger signals before trading.",
                "type": "neutral"})

        # Volume analysis
        high_vol = [(s, d) for s, d in pair_data.items() if d.get('volume', 0) > 50_000_000 and abs(d.get('change', 0)) > 2]
        if high_vol:
            top = sorted(high_vol, key=lambda x: x[1]['volume'], reverse=True)[0]
            sym, data = top
            name = sym.replace('USDT', '')
            insights.append({"title": f"High Volume Alert: {name}",
                "body": f"{name} showing ${data['volume']/1e6:.0f}M in 24h volume with {data['change']:+.1f}% price move. High volume moves are more likely to sustain direction. {'Bot has this pair active.' if sym in analysis_pairs else 'Not currently in bot — consider adding via Railway TRADING_PAIRS.'}",
                "type": "bullish" if data['change'] > 0 else "bearish"})

        # Build watchlist from candidates
        wl_candidates = [(s, d) for s, d in pair_data.items() if s in watch_candidates and d.get('rsi')]
        wl_candidates.sort(key=lambda x: x[1]['change'], reverse=True)
        for sym, data in wl_candidates[:3]:
            name = sym.replace('USDT', '')
            rsi = data['rsi']
            change = data['change']
            if rsi < 40:
                signal = "buy"
                reason = f"RSI oversold at {rsi} with {change:+.1f}% 24h move — potential bounce candidate."
            elif rsi > 65:
                signal = "avoid"
                reason = f"RSI at {rsi} — overbought. Wait for pullback before entry."
            else:
                signal = "watch"
                reason = f"RSI at {rsi}, {change:+.1f}% in 24h — neutral setup, monitor for breakout."
            watchlist.append({"symbol": sym, "name": name, "reason": reason, "signal": signal})

        # Pair recommendations based on volume + trend
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
                        "reason": f"RSI {d['rsi']} overbought — wait for pullback before adding."})
                else:
                    pair_recs.append({"symbol": sym, "name": name, "action": "keep" if in_bot else "add",
                        "reason": f"{d['change']:+.1f}% 24h, RSI {d['rsi']} — solid mid-cap with good liquidity on Binance."})
            if len(pair_recs) >= 3:
                break

        # Risk warning
        losers = [(s, d) for s, d in pair_data.items() if s in analysis_pairs and d.get('change', 0) < -5]
        if losers:
            names = ', '.join([s.replace('USDT','') for s,_ in losers])
            risk_warning = f"{names} down over 5% today — check your OCO stop loss orders are active in Binance app."
        elif regime == 'bearish':
            risk_warning = "Bear market regime detected — bot is using tighter RSI thresholds and lower take-profit targets. Reduce position sizes if uncertain."
        elif btc_rsi and btc_rsi > 75:
            risk_warning = f"BTC RSI at {btc_rsi} — historically high. Consider reducing exposure or tightening stop losses on open positions."
        else:
            risk_warning = "No major risk signals detected. Ensure OCO orders are active in Binance for all open positions."

    except Exception as e:
        log.warning(f"Insights data fetch error: {e}")
        insights = [{"title": "Market data temporarily unavailable", "body": "Could not fetch live Binance data. Check Railway logs for details.", "type": "warning"}]
        watchlist = []
        pair_recs = []
        risk_warning = "Check Railway logs — insights data fetch failed."

    # Try Gemini AI (free, works globally) using the live data we just fetched
    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    ai_powered = False
    if gemini_key and pair_data:
        try:
            import json
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


@app.route('/api/stats')
def get_stats():
    """Per-pair win rate, PnL calendar, and portfolio allocation"""
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

        # PnL calendar — last 30 days
        from datetime import datetime, timedelta
        calendar = {}
        for i in range(30):
            d = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            calendar[d] = {'pnl': 0.0, 'trades': 0}
        for t in trades:
            if t.get('side') == 'sell':
                d = t.get('date', '')
                if d in calendar:
                    pnl = t.get('pnl', 0) or 0
                    calendar[d]['pnl'] = round(calendar[d]['pnl'] + pnl, 2)
                    calendar[d]['trades'] += 1
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

    approved, reason = risk_manager.check_trade(pair, action, confidence)
    if not approved:
        return jsonify({'error': f'Risk manager blocked: {reason}'}), 400

    is_manual = data.get('manual', False)
    if is_manual:
        log.info(f"Manual trade: {action} {pair}")
    try:
        result = trader.execute_trade(pair, action, config.max_trade_pct)
        risk_manager.record_trade(pair)  # Start cooldown after trade
        send_telegram(
            f"{'🟢' if action == 'buy' else '🔴'} *{action.upper()} {pair}*\n"
            f"Confidence: {confidence}%\n"
            f"Market: {signal_engine.market_regime if signal_engine else 'unknown'}\n"
            f"Order: {result.get('orderId', 'N/A')}"
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        risk_manager.release_lock(pair)  # Always release lock


@app.route('/api/history')
def get_history():
    try:
        if trader:
            return jsonify({'trades': trader.get_real_trade_history()})
        return jsonify({'trades': config.load_trade_history()})
    except Exception as e:
        return jsonify({'trades': [], 'error': str(e)})


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

@app.route('/api/sniper/test')
def test_sniper():
    """Test endpoint — verify sniper is working correctly"""
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


@app.route('/api/sniper/toggle', methods=['POST'])
def toggle_sniper():
    if not sniper:
        return jsonify({'error': 'Not initialised'}), 400
    sniper.active = request.json.get('active', False)
    config.sniper_active = sniper.active
    config.save()
    return jsonify({'active': sniper.active})


@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        config.update(request.json)
        config.save()
        if risk_manager:
            risk_manager.reload(config)
        return jsonify({'success': True})
    return jsonify(config.to_dict())


if __name__ == '__main__':
    init_trader()
    port = int(os.environ.get('PORT', 8080))
    log.info(f"AutoTrader Pro starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
