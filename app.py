"""
AutoTrader Pro - Main Flask Server
Added: market regime endpoint
"""

from flask import Flask, jsonify, request, send_file
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

    # Static intelligent fallback — always works
    fallback = {
        "regime": regime,
        "updated": datetime.now().strftime("%H:%M"),
        "insights": [
            {"title": "Market Regime: " + regime.upper(), "body": "AI has detected a " + regime + " market. Take profit is auto-set to " + str(regime_tp) + "% for all trades. RSI thresholds auto-adjusted accordingly.", "type": regime if regime in ["bullish","bearish","warning"] else "neutral"},
            {"title": "Per-Pair RSI Active", "body": "BTC/ETH using 25/80 thresholds. SOL/LINK/BNB using 30/75. ARB/RENDER and new pairs using 32/80 for more selective entries.", "type": "neutral"},
            {"title": "Trade Cooldown Running", "body": "60 minute cooldown after every trade prevents overbuying. Sell signals are skipped if no holdings exist for that pair.", "type": "neutral"}
        ],
        "watchlist": [
            {"symbol": "BTCUSDT", "name": "Bitcoin", "reason": "Lead indicator for entire market. BTC RSI at current levels suggests watching for oversold entry.", "signal": "watch"},
            {"symbol": "SOLUSDT", "name": "Solana", "reason": "Strong ecosystem fundamentals, reliable RSI signals on dips.", "signal": "watch"},
            {"symbol": "ARBUSDT", "name": "Arbitrum", "reason": "L2 narrative strong. High volume on Binance, good for RSI-based entries.", "signal": "watch"}
        ],
        "pair_recommendations": [
            {"symbol": "XRPUSDT", "name": "XRP", "action": "add", "reason": "High liquidity on Binance, responds well to RSI signals. Mid-cap tier (30/75)."},
            {"symbol": "DOGEUSDT", "name": "Dogecoin", "action": "keep" if "DOGEUSDT" in getattr(config, "trading_pairs", []) else "avoid", "reason": "High volume but volatile — only suitable in bull markets with tight stop loss."},
            {"symbol": "AVAXUSDT", "name": "Avalanche", "action": "add", "reason": "Strong DeFi ecosystem, good technical signals. Would auto-assign 32/80 RSI tier."}
        ],
        "risk_warning": "Verify OCO stop loss orders are active in Binance app for all open positions before leaving the bot unattended."
    }

    try:
        import requests as req_lib, json
        current_pairs = ','.join([p.replace('USDT','') for p in getattr(config, 'trading_pairs', ['BTC','ETH','BNB','SOL','RENDER','LINK','ARB'])])
        prompt = (
            "You are a crypto market analyst for a Binance spot trading bot. Date: " + datetime.now().strftime('%Y-%m-%d') + ". "
            "Market regime: " + regime + ". Auto take profit: " + str(regime_tp) + "%. "
            "Currently trading: " + current_pairs + ". "
            "Return ONLY valid JSON, no other text: "
            '{"insights":[{"title":"str","body":"2 sentences","type":"bullish|bearish|neutral|warning"},'
            '{"title":"str","body":"2 sentences","type":"bullish|bearish|neutral|warning"},'
            '{"title":"str","body":"2 sentences","type":"bullish|bearish|neutral|warning"}],'
            '"watchlist":[{"symbol":"BTCUSDT","name":"Bitcoin","reason":"1 sentence","signal":"buy|watch|avoid"},'
            '{"symbol":"ETHUSDT","name":"Ethereum","reason":"1 sentence","signal":"buy|watch|avoid"},'
            '{"symbol":"SOLUSDT","name":"Solana","reason":"1 sentence","signal":"buy|watch|avoid"}],'
            '"pair_recommendations":[{"symbol":"XRPUSDT","name":"XRP","action":"add|remove|keep","reason":"1 sentence"},'
            '{"symbol":"AVAXUSDT","name":"Avalanche","action":"add|remove|keep","reason":"1 sentence"},'
            '{"symbol":"DOGEUSDT","name":"Dogecoin","action":"add|remove|keep","reason":"1 sentence"}],'
            '"risk_warning":"1 sentence risk warning"}'
        )
        response = req_lib.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 1000,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=25
        )
        if response.status_code == 200:
            data = response.json()
            text = data["content"][0]["text"].strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(text[start:end])
                result["regime"] = regime
                result["updated"] = datetime.now().strftime("%H:%M")
                result.setdefault("pair_recommendations", fallback["pair_recommendations"])
                return jsonify(result)
    except Exception as e:
        log.debug(f"AI insights unavailable: {e}")

    return jsonify(fallback)


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

    try:
        try:
            result = trader.execute_trade(pair, action, config.max_trade_pct)
        finally:
            risk_manager.release_lock(pair)  # Always release lock
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
        return jsonify({'active': False, 'detections': []})
    return jsonify({'active': sniper.active, 'detections': sniper.recent_detections[-10:]})


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
