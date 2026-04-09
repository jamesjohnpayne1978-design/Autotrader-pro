"""
AutoTrader Pro - Main Flask Server
Optimised for Railway.app deployment
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import threading
import logging
import os
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


def init_trader():
    global trader, sniper, signal_engine
    if config.api_key and config.api_secret:
        try:
            trader = Trader(config)
            signal_engine = SignalEngine(config, trader)
            sniper = ListingSniper(config, trader, risk_manager)
            if config.sniper_active:
                sniper_thread = threading.Thread(target=sniper.run, daemon=True)
                sniper_thread.start()
            signal_thread = threading.Thread(target=signal_engine.run, daemon=True)
            signal_thread.start()
            log.info("Trader, Sniper and Signal Engine initialised.")
        except Exception as e:
            log.error(f"Failed to initialise trader: {e}")


# ─── DASHBOARD ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_file('index.html')


# ─── HEALTH ────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'connected': trader is not None,
        'sniper': sniper.active if sniper else False,
        'version': '1.0.0'
    })


# ─── CONFIG ────────────────────────────────────────────────────────
@app.route('/api/config', methods=['POST'])
def set_config():
    data = request.json
    config.api_key = data.get('api_key', '')
    config.api_secret = data.get('api_secret', '')
    config.save()
    init_trader()
    return jsonify({'success': True})


# ─── PORTFOLIO ─────────────────────────────────────────────────────
@app.route('/api/portfolio')
def get_portfolio():
    if not trader:
        return jsonify({'error': 'Not connected'}), 400
    try:
        return jsonify(trader.get_portfolio())
    except Exception as e:
        log.error(f"Portfolio error: {e}")
        return jsonify({'error': str(e)}), 500


# ─── PRICES ────────────────────────────────────────────────────────
@app.route('/api/prices')
def get_prices():
    if not trader:
        return jsonify({'error': 'Not connected'}), 400
    try:
        return jsonify({'pairs': trader.get_prices()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── SIGNALS ───────────────────────────────────────────────────────
@app.route('/api/signals')
def get_signals():
    if not signal_engine:
        return jsonify({'signals': []})
    try:
        return jsonify({'signals': signal_engine.get_latest_signals()})
    except Exception as e:
        return jsonify({'signals': [], 'error': str(e)})


# ─── EXECUTE TRADE ─────────────────────────────────────────────────
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
        result = trader.execute_trade(pair, action, config.max_trade_pct)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── HISTORY ───────────────────────────────────────────────────────
@app.route('/api/history')
def get_history():
    try:
        return jsonify({'trades': config.load_trade_history()})
    except Exception as e:
        return jsonify({'trades': [], 'error': str(e)})


# ─── SNIPER ────────────────────────────────────────────────────────
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


# ─── SETTINGS ──────────────────────────────────────────────────────
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
