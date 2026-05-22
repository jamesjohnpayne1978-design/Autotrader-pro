"""
AutoTrader Pro - New Listing Sniper
Detects new Binance listings via exchange info polling (most reliable method)

NOTE: The sniper deliberately bypasses the global `approval_mode` setting.
Snipes are time-sensitive (listing pumps last minutes), so waiting for human
approval defeats the purpose of having a sniper. AI signal trades still respect
approval_mode normally. To pause sniping entirely, turn off the Sniper toggle.
"""

import time
import logging
import json
import requests
from datetime import datetime
from threading import Thread

log = logging.getLogger(__name__)

# Binance public API - no scraping needed
BINANCE_API = "https://api.binance.com/api/v3"
BINANCE_NEWS_API = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&pageNo=1&pageSize=20"
BINANCE_NEWS_API_V2 = "https://www.binance.com/en/support/announcement/new-cryptocurrency-listing?c=48&navId=48"


def _sniper_budget(config):
    """Read sniper budget defensively - config key has been spelled both
    `sniper_budget_usdt` and `sniper_budget` in different versions, and we
    fall back to a sane default if neither exists."""
    val = getattr(config, 'sniper_budget_usdt', None)
    if val is None:
        val = getattr(config, 'sniper_budget', None)
    if val is None:
        val = 50.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 50.0


class ListingSniper:
    def __init__(self, config, trader, risk_manager):
        self.config = config
        self.trader = trader
        self.risk_manager = risk_manager
        self.active = config.sniper_active
        self.seen_symbols = set()
        self.seen_news_ids = set()
        self.recent_detections = []
        self._load_seen()
        log.info("Sniper initialised - watching for new listings")

    def _load_seen(self):
        try:
            with open('/data/seen_listings.json', 'r') as f:
                data = json.load(f)
                self.seen_symbols = set(data.get('symbols', []))
                self.seen_news_ids = set(data.get('news_ids', []))
        except Exception:
            self.seen_symbols = set()
            self.seen_news_ids = set()

    def _save_seen(self):
        try:
            with open('/data/seen_listings.json', 'w') as f:
                json.dump({
                    'symbols': list(self.seen_symbols),
                    'news_ids': list(self.seen_news_ids)
                }, f)
        except Exception:
            pass

    def run(self):
        log.info("Sniper thread started - seeding known symbols...")
        # Always force-seed on startup to get current state
        self._seed_known_symbols(force=True)
        log.info(f"Sniper ready - watching {len(self.seen_symbols)} existing pairs for new listings")
        check_count = 0
        while True:
            if self.active:
                try:
                    self._check_new_exchange_symbols()
                    # Check news API every 4 cycles (~60s)
                    if check_count % 4 == 0:
                        self._check_binance_news()
                    if check_count % 20 == 0:  # Log heartbeat every 5 mins
                        log.info(f"Sniper heartbeat - watching {len(self.seen_symbols)} pairs, {len(self.recent_detections)} detections so far")
                    check_count += 1
                except Exception as e:
                    log.error(f"Sniper check error: {e}")
            time.sleep(15)

    def _seed_known_symbols(self, force=False):
        """Load all current symbols so we only alert on NEW ones"""
        try:
            info = self.trader.client.get_exchange_info()
            current = {s['symbol'] for s in info['symbols']
                      if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'}
            if force or not self.seen_symbols:
                # Always force-seed on startup - never trust stale saved state
                self.seen_symbols = current
                self._save_seen()
                log.info(f"Sniper seeded with {len(current)} existing USDT pairs (force={force})")
            else:
                log.info(f"Sniper using {len(self.seen_symbols)} saved symbols")
        except Exception as e:
            log.warning(f"Could not seed symbols: {e}")

    def _check_new_exchange_symbols(self):
        """Most reliable method - directly polls Binance exchange info"""
        try:
            info = self.trader.client.get_exchange_info()
            current = {s['symbol'] for s in info['symbols']
                      if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'}

            if not self.seen_symbols:
                self.seen_symbols = current
                self._save_seen()
                return

            new_symbols = current - self.seen_symbols
            if new_symbols:
                for symbol in new_symbols:
                    # Filter out stablecoin pairs and obvious test tokens
                    base = symbol.replace('USDT', '')
                    if any(x in base for x in ['USD', 'EUR', 'GBP', 'BUSD', 'USDC', 'TUSD', 'TEST']):
                        continue
                    log.info(f"🎯 NEW SYMBOL DETECTED: {symbol}")
                    self._handle_new_listing(symbol, f"New pair listed on Binance: {symbol}", 'exchange_api')
                self.seen_symbols = current
                self._save_seen()
        except Exception as e:
            log.debug(f"Exchange check error: {e}")

    def _check_binance_news(self):
        """Check Binance official news API for listing announcements"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            resp = requests.get(BINANCE_NEWS_API, headers=headers, timeout=10)
            if resp.status_code != 200:
                return

            data = resp.json()
            articles = data.get('data', {}).get('articles', [])

            for article in articles:
                article_id = str(article.get('id', ''))
                title = article.get('title', '')

                if article_id in self.seen_news_ids:
                    continue

                self.seen_news_ids.add(article_id)

                # Check if it's a listing announcement
                title_lower = title.lower()
                if any(kw in title_lower for kw in ['will list', 'lists', 'listing', 'new listing', 'adds']):
                    symbol = self._extract_symbol(title)
                    if symbol and symbol not in self.seen_symbols:
                        log.info(f"📰 LISTING ANNOUNCEMENT: {title} → {symbol}")
                        self._handle_new_listing(symbol, title, 'news_api')

            # Keep seen_news_ids manageable
            if len(self.seen_news_ids) > 500:
                self.seen_news_ids = set(list(self.seen_news_ids)[-200:])
            self._save_seen()

        except Exception as e:
            log.debug(f"News check error: {e}")

    def _extract_symbol(self, title: str):
        """Extract coin symbol from announcement title"""
        import re
        SKIP = {'USD','USDT','USDC','BUSD','EUR','GBP','BTC','ETH','BNB',
                'THE','FOR','AND','NEW','OUR','ITS','NOT','ALL','ANY'}
        patterns = [
            r'\(([A-Z]{2,12})(?:/USDT|/BTC|/ETH)?\)',          # (TOKEN) or (TOKEN/USDT)
            r'(?:will\s+list|lists|listing|adds?)\s+([A-Z]{2,12})',  # will list TOKEN
            r'([A-Z]{3,12})\s+(?:token|coin)?\s+(?:to|on)\s+Binance', # TOKEN to Binance
            r'Binance\s+(?:Lists|Will\s+List)\s+([A-Z]{2,12})',  # Binance Lists TOKEN
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, title, re.IGNORECASE):
                coin = match.group(1).upper()
                if coin not in SKIP and len(coin) >= 2:
                    symbol = coin + 'USDT'
                    log.info(f"Extracted symbol {symbol} from: {title[:60]}")
                    return symbol
        return None

    def _handle_new_listing(self, symbol: str, title: str, source: str):
        """Process a detected new listing"""
        detection = {
            'symbol': symbol,
            'title': title[:100],
            'source': source,
            'time': datetime.now().strftime('%H:%M:%S'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'status': 'detected',
            'action': 'Detected - evaluating...'
        }
        self.recent_detections.insert(0, detection)
        log.info(f"Processing new listing: {symbol} via {source}")

        # Risk check
        approved, reason = self.risk_manager.check_snipe(symbol)
        if not approved:
            detection['status'] = 'blocked'
            detection['action'] = f'Risk blocked: {reason}'
            log.info(f"Sniper blocked {symbol}: {reason}")
            return

        # NOTE: Sniper intentionally bypasses the global `approval_mode` setting.
        # Listing pumps happen within seconds and waiting for human approval is
        # too slow. To pause sniper buying, toggle off the Sniper switch on the
        # dashboard. AI signal trades still respect approval_mode separately.
        budget = _sniper_budget(self.config)
        detection['status'] = 'buying'
        detection['action'] = f'Auto-buying ${budget:.0f}'
        self._send_telegram_buying(symbol, title, budget)
        t = Thread(target=self._execute_snipe, args=(symbol, detection, budget), daemon=True)
        t.start()

    def _execute_snipe(self, symbol: str, detection: dict, budget: float):
        """Execute the snipe trade, then place an OCO order at Binance to handle
        TP and SL at the exchange level. No more 30-min force-sell - position
        runs until OCO triggers OR the user manually closes."""
        try:
            # Wait up to 30s for trading to open
            for _ in range(30):
                try:
                    self.trader.client.get_symbol_ticker(symbol=symbol)
                    break
                except Exception:
                    time.sleep(1)

            result = self.trader.snipe_listing(symbol, budget)
            if result.get('success'):
                buy_price = result.get('fill_price', 0)
                qty = result.get('quantity', 0)
                tp_pct = float(self.config.sniper_tp_pct)
                sl_pct = float(self.config.sniper_sl_pct)

                # Capture fill details for the dashboard
                detection['status'] = 'bought'
                detection['action'] = f"Bought ${budget:.0f}"
                detection['buy_price'] = buy_price
                detection['qty'] = qty
                detection['usdt_spent'] = result.get('usdt_spent', budget)
                detection['bought_at'] = datetime.now().isoformat(timespec='seconds')
                detection['tp_pct'] = tp_pct
                detection['sl_pct'] = sl_pct
                log.info(f"Snipe executed: {symbol} @ ${buy_price:.6f} qty={qty}")
                self._send_telegram_bought(symbol, budget, result)

                # Place OCO order at Binance - this is now the primary exit mechanism
                # rather than a 30-min in-memory force-sell. Position survives bot
                # restarts because the OCO lives at the exchange.
                try:
                    # Give Binance ~1s to settle the buy before placing the OCO
                    time.sleep(1)
                    oco_result = self.trader.place_snipe_oco(symbol, qty, buy_price, tp_pct, sl_pct)
                    if oco_result.get('success'):
                        detection['oco_placed'] = True
                        detection['oco_tp_price'] = oco_result.get('tp_price')
                        detection['oco_sl_price'] = oco_result.get('sl_price')
                        log.info(f"Snipe OCO active: TP ${oco_result.get('tp_price')} SL ${oco_result.get('sl_price')}")
                        self._telegram_post(
                            f"🛡️ *Snipe protected by OCO*\n`{symbol}`\n"
                            f"TP: ${oco_result.get('tp_price')} (+{tp_pct}%)\n"
                            f"SL: ${oco_result.get('sl_price')} (-{sl_pct}%)\n"
                            f"_Position runs until TP or SL hits - no 30-min force sell._"
                        )
                    else:
                        detection['oco_placed'] = False
                        detection['oco_error'] = oco_result.get('error', 'unknown')
                        log.warning(f"OCO failed for {symbol}: {detection['oco_error']}. Falling back to monitor.")
                        self._telegram_post(
                            f"⚠️ *Snipe OCO failed for {symbol}*\n"
                            f"Reason: {detection['oco_error'][:120]}\n"
                            f"_Position is UNPROTECTED - sell manually when ready._"
                        )
                except Exception as e:
                    detection['oco_placed'] = False
                    detection['oco_error'] = str(e)
                    log.error(f"OCO placement exception for {symbol}: {e}")

                # Light monitor thread for dashboard live data only (no force-sell)
                Thread(target=self._monitor_position, args=(symbol, detection), daemon=True).start()
            else:
                detection['status'] = 'failed'
                err = result.get('error', 'unknown')
                detection['action'] = f"Failed: {err}"
                log.error(f"Snipe failed for {symbol}: {err}")
                self._send_telegram_failed(symbol, err)
        except Exception as e:
            detection['status'] = 'error'
            detection['action'] = str(e)
            log.error(f"Snipe error for {symbol}: {e}")
            self._send_telegram_failed(symbol, str(e))

    def _monitor_position(self, symbol: str, detection: dict = None):
        """Watches a sniped position for dashboard purposes - updates current
        price and P&L in the detection dict every few seconds. The actual exit
        (TP/SL) is handled by the OCO order at Binance, not by this thread.

        Stops when the position is sold (balance goes to ~0) or after 24 hours
        as a safety bound. No force-sell."""
        buy_price = detection.get('buy_price') if detection else None
        tp_pct = (detection.get('tp_pct') if detection else None) or float(self.config.sniper_tp_pct)
        sl_pct = (detection.get('sl_pct') if detection else None) or float(self.config.sniper_sl_pct)
        start = time.time()
        max_watch_seconds = 24 * 60 * 60  # safety: stop watching after 24h
        base = symbol.replace('USDT', '')

        while time.time() - start < max_watch_seconds:
            try:
                price = float(self.trader.client.get_symbol_ticker(symbol=symbol)['price'])
                if buy_price is None or buy_price <= 0:
                    buy_price = price
                    if detection is not None:
                        detection['buy_price'] = price
                    continue

                change_pct = ((price - buy_price) / buy_price) * 100
                if detection is not None:
                    detection['current_price'] = price
                    detection['change_pct'] = round(change_pct, 2)
                    detection['unrealised_usdt'] = round((price - buy_price) * detection.get('qty', 0), 2)
                    detection['monitoring'] = True

                # Check if position has been closed (OCO fired or manual sell)
                # If we have <5% of original quantity, position is essentially closed
                try:
                    account = self.trader.client.get_account()
                    current_qty = next(
                        (float(b['free']) + float(b['locked']) for b in account['balances'] if b['asset'] == base),
                        0.0
                    )
                    original_qty = detection.get('qty', 0) if detection else 0
                    if original_qty > 0 and current_qty < original_qty * 0.05:
                        # Position closed - determine exit type by comparing to TP/SL prices
                        if detection is not None:
                            if change_pct >= tp_pct * 0.9:  # within 10% of TP target
                                detection['status'] = 'tp_hit'
                                detection['action'] = f"TP filled +{change_pct:.1f}%"
                            elif change_pct <= -sl_pct * 0.9:
                                detection['status'] = 'sl_hit'
                                detection['action'] = f"SL filled {change_pct:.1f}%"
                            else:
                                detection['status'] = 'closed'
                                detection['action'] = f"Closed {change_pct:+.1f}%"
                            detection['monitoring'] = False
                        log.info(f"Snipe position closed: {symbol} at {change_pct:+.1f}%")
                        self._telegram_post(
                            f"✅ *Snipe closed*\n`{symbol}`\nResult: {change_pct:+.1f}%"
                        )
                        return
                except Exception:
                    pass

            except Exception as e:
                log.debug(f"Monitor error {symbol}: {e}")
            time.sleep(15)  # Check every 15s - no rush since OCO does the actual work

        if detection is not None:
            detection['monitoring'] = False

    def _telegram_post(self, text: str):
        """Internal helper - actually sends the message. Silent on failure."""
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                json={'chat_id': str(self.config.telegram_chat_id), 'text': text, 'parse_mode': 'Markdown'},
                timeout=5
            )
        except Exception as e:
            log.debug(f"Telegram failed: {e}")

    def _send_telegram_buying(self, symbol: str, title: str, budget: float):
        """Alert: sniper detected a listing and is attempting to buy."""
        msg = (
            f"🎯 *NEW LISTING DETECTED*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Source: {title[:80]}\n\n"
            f"Attempting buy: ${budget:.0f}\n"
            f"TP: +{self.config.sniper_tp_pct}% · SL: -{self.config.sniper_sl_pct}%\n"
            f"Max hold: 30 min"
        )
        self._telegram_post(msg)

    def _send_telegram_bought(self, symbol: str, budget: float, result: dict):
        """Alert: sniper successfully bought the listing."""
        msg = (
            f"✅ *SNIPE EXECUTED*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Size: ${budget:.0f}\n\n"
            f"Now monitoring for TP/SL..."
        )
        self._telegram_post(msg)

    def _send_telegram_failed(self, symbol: str, error: str):
        """Alert: sniper buy attempt failed."""
        msg = (
            f"❌ *SNIPE FAILED*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Reason: {str(error)[:200]}"
        )
        self._telegram_post(msg)

    # Kept for backward compatibility in case other code still calls it
    def _send_telegram(self, symbol: str, title: str):
        self._send_telegram_buying(symbol, title, _sniper_budget(self.config))
