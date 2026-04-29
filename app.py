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

    try:
        import requests as req_lib, json

        # Get current prices for context
        price_context = ""
        if trader:
            try:
                prices = trader.get_prices()
                price_lines = []
                for p in prices:
                    price_lines.append(f"{p['symbol']}: ${p['price']} ({p['change']:+.1f}% 24h)")
                price_context = "\n".join(price_lines)
            except Exception:
                pass

        current_pairs = ','.join([p.replace('USDT','') for p in getattr(config, 'trading_pairs', ['BTC','ETH','BNB','SOL','RENDER','LINK','ARB'])])
        today = datetime.now().strftime('%Y-%m-%d %H:%M')

        prompt = (
            "You are a professional crypto trading analyst with access to real-time market data. "
            "Today is " + today + ". Market regime detected: " + regime + ". Auto take-profit: " + str(regime_tp) + "%. "
            "Bot is currently trading: " + current_pairs + ". "
            "Current live prices:\n" + price_context + "\n\n"
            "Use your knowledge of current crypto market conditions, recent news, BTC dominance trends, "
            "macro factors (Fed rates, ETF flows, institutional activity), and on-chain signals to provide "
            "SPECIFIC, ACTIONABLE, and CURRENT analysis. Do NOT give generic advice. "
            "Reference actual current market conditions, specific price levels, and real catalysts. "
            "For pair recommendations, suggest specific coins that are genuinely interesting RIGHT NOW "
            "based on current narratives (e.g. AI tokens, L2s, DeFi, memecoins, RWA, etc). "
            "Be direct and specific like a professional trader would be. "
            "Return ONLY this JSON with no other text:\n"
            '{"insights":['
            '{"title":"string under 8 words","body":"2-3 specific sentences with real data points","type":"bullish|bearish|neutral|warning"},'
            '{"title":"string under 8 words","body":"2-3 specific sentences with real data points","type":"bullish|bearish|neutral|warning"},'
            '{"title":"string under 8 words","body":"2-3 specific sentences with real data points","type":"bullish|bearish|neutral|warning"}'
            '],'
            '"watchlist":['
            '{"symbol":"XYZUSDT","name":"Name","reason":"1 specific sentence with current catalyst","signal":"buy|watch|avoid"},'
            '{"symbol":"XYZUSDT","name":"Name","reason":"1 specific sentence with current catalyst","signal":"buy|watch|avoid"},'
            '{"symbol":"XYZUSDT","name":"Name","reason":"1 specific sentence with current catalyst","signal":"buy|watch|avoid"}'
            '],'
            '"pair_recommendations":['
            '{"symbol":"XYZUSDT","name":"Name","action":"add|remove|keep","reason":"specific current reason"},'
            '{"symbol":"XYZUSDT","name":"Name","action":"add|remove|keep","reason":"specific current reason"},'
            '{"symbol":"XYZUSDT","name":"Name","action":"add|remove|keep","reason":"specific current reason"}'
            '],'
            '"risk_warning":"1 specific sentence about a real current risk to watch"}'
        )

        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise Exception("No ANTHROPIC_API_KEY set")

        response = req_lib.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 1500,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=45
        )

        if response.status_code == 200:
            data = response.json()
            # Extract text from content blocks (may include tool use blocks)
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            text = text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(text[start:end])
                result["regime"] = regime
                result["updated"] = datetime.now().strftime("%H:%M")
                result["ai_powered"] = True
                result.setdefault("pair_recommendations", [])
                log.info("AI insights generated successfully with web search")
                return jsonify(result)
            else:
                log.warning(f"Could not parse AI response: {text[:200]}")
        else:
            log.warning(f"Anthropic API error: {response.status_code} {response.text[:200]}")

    except Exception as e:
        log.warning(f"AI insights failed: {e}")

    # Fallback — still useful but clearly marked
    return jsonify({
        "regime": regime,
        "updated": datetime.now().strftime("%H:%M"),
        "ai_powered": False,
        "insights": [
            {"title": "Add API Key For Live AI", "body": "Add ANTHROPIC_API_KEY to Railway Variables to unlock real-time AI market analysis with web search. The bot will then pull live news, price action and macro data to generate fresh insights every time you refresh.", "type": "warning"},
            {"title": "Market Regime: " + regime.upper(), "body": "Bot has detected a " + regime + " market. Take profit auto-set to " + str(regime_tp) + "%. RSI thresholds adjusted per tier — BTC/ETH 25/80, mid caps 30/75, small caps 32/80.", "type": "neutral"},
            {"title": "Strategy Update Active", "body": "Confidence threshold raised to 72%. Trend filter now blocks buys when price is more than 1.5% below 50MA — prevents buying into falling coins.", "type": "neutral"}
        ],
        "watchlist": [
            {"symbol": "BTCUSDT", "name": "Bitcoin", "reason": "Add API key for live AI watchlist recommendations based on current market data.", "signal": "watch"},
            {"symbol": "ETHUSDT", "name": "Ethereum", "reason": "Add ANTHROPIC_API_KEY to Railway for fresh AI-driven watchlist every refresh.", "signal": "watch"},
            {"symbol": "SOLUSDT", "name": "Solana", "reason": "Live AI analysis will suggest specific coins based on current narratives and catalysts.", "signal": "watch"}
        ],
        "pair_recommendations": [
            {"symbol": "ANTHROPIC_API_KEY", "name": "Setup Required", "action": "add", "reason": "Go to Railway → Variables → add ANTHROPIC_API_KEY to get real AI pair recommendations"},
            {"symbol": "AVAXUSDT", "name": "Avalanche", "action": "add", "reason": "Strong DeFi ecosystem, auto-assigned 30/75 RSI tier as mid-cap."},
            {"symbol": "RENDERUSDT", "name": "Render", "action": "keep", "reason": "AI/GPU narrative remains strong. Currently on 32/80 small-cap RSI tier."}
        ],
        "risk_warning": "Add ANTHROPIC_API_KEY to Railway Variables for live AI risk analysis based on current market conditions."
    })


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
