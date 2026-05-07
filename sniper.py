"""
AutoTrader Pro - New Listing Sniper
Detects new Binance listings via exchange info polling (most reliable method)
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
        log.info("Sniper initialised — watching for new listings")

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
        log.info("Sniper thread started — seeding known symbols...")
        # Always force-seed on startup to get current state
        self._seed_known_symbols(force=True)
        log.info(f"Sniper ready — watching {len(self.seen_symbols)} existing pairs for new listings")
        check_count = 0
        while True:
            if self.active:
                try:
                    self._check_new_exchange_symbols()
                    # Check news API every 4 cycles (~60s)
                    if check_count % 4 == 0:
                        self._check_binance_news()
                    if check_count % 20 == 0:  # Log heartbeat every 5 mins
                        log.info(f"Sniper heartbeat — watching {len(self.seen_symbols)} pairs, {len(self.recent_detections)} detections so far")
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
                # Always force-seed on startup — never trust stale saved state
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
            'action': 'Detected — evaluating...'
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

        # Approval mode — log and notify only
        if self.config.approval_mode:
            detection['status'] = 'pending'
            detection['action'] = 'Waiting — approval mode ON'
            self._send_telegram(symbol, title)
            log.info(f"Sniper found {symbol} — approval mode ON, not auto-buying")
            return

        # Auto mode — execute in background thread
        detection['status'] = 'buying'
        detection['action'] = f'Auto-buying ${self.config.sniper_budget_usdt}'
        t = Thread(target=self._execute_snipe, args=(symbol, detection), daemon=True)
        t.start()

    def _execute_snipe(self, symbol: str, detection: dict):
        """Execute the snipe trade"""
        try:
            # Wait up to 30s for trading to open
            for _ in range(30):
                try:
                    self.trader.client.get_symbol_ticker(symbol=symbol)
                    break
                except Exception:
                    time.sleep(1)

            result = self.trader.snipe_listing(symbol, self.config.sniper_budget_usdt)
            if result.get('success'):
                detection['status'] = 'bought'
                detection['action'] = f"Bought ${self.config.sniper_budget_usdt:.0f}"
                log.info(f"Snipe executed: {symbol}")
                # Monitor in background
                Thread(target=self._monitor_position, args=(symbol,), daemon=True).start()
            else:
                detection['status'] = 'failed'
                detection['action'] = f"Failed: {result.get('error', 'unknown')}"
        except Exception as e:
            detection['status'] = 'error'
            detection['action'] = str(e)
            log.error(f"Snipe error for {symbol}: {e}")

    def _monitor_position(self, symbol: str):
        """Monitor sniped position for TP/SL"""
        tp = self.config.sniper_tp_pct / 100
        sl = self.config.sniper_sl_pct / 100
        buy_price = None
        start = time.time()
        max_hold = 30 * 60  # 30 minutes

        while time.time() - start < max_hold:
            try:
                price = float(self.trader.client.get_symbol_ticker(symbol=symbol)['price'])
                if buy_price is None:
                    buy_price = price
                    continue
                change = (price - buy_price) / buy_price
                if change >= tp:
                    log.info(f"Snipe TP hit {symbol}: +{change*100:.1f}%")
                    self.trader.execute_trade(f"{symbol.replace('USDT','/USDT')}", 'sell', 100)
                    return
                if change <= -sl:
                    log.info(f"Snipe SL hit {symbol}: {change*100:.1f}%")
                    self.trader.execute_trade(f"{symbol.replace('USDT','/USDT')}", 'sell', 100)
                    return
            except Exception as e:
                log.debug(f"Monitor error {symbol}: {e}")
            time.sleep(5)

        # Time limit — force sell
        log.info(f"Snipe time limit {symbol} — selling")
        try:
            self.trader.execute_trade(f"{symbol.replace('USDT','/USDT')}", 'sell', 100)
        except Exception as e:
            log.error(f"Force sell failed: {e}")

    def _send_telegram(self, symbol: str, title: str):
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            return
        try:
            msg = (
                f"🎯 *New Listing Detected!*\n\n"
                f"Symbol: `{symbol}`\n"
                f"Source: {title[:80]}\n\n"
                f"Budget: ${self.config.sniper_budget_usdt} | "
                f"TP: +{self.config.sniper_tp_pct}% | SL: -{self.config.sniper_sl_pct}%\n\n"
                f"⚠️ Approval mode ON — not auto-buying"
            )
            requests.post(
                f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                json={'chat_id': self.config.telegram_chat_id, 'text': msg, 'parse_mode': 'Markdown'},
                timeout=5
            )
        except Exception as e:
            log.debug(f"Telegram failed: {e}")
