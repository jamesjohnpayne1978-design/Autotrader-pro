"""
AutoTrader Pro - New Listing Sniper
Monitors Binance for new coin listings and executes sniping strategy
"""

import time
import logging
import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup
from binance.client import Client

log = logging.getLogger(__name__)

BINANCE_ANNOUNCEMENTS = "https://www.binance.com/en/support/announcement/new-cryptocurrency-listing"


class ListingSniper:
    def __init__(self, config, trader, risk_manager):
        self.config = config
        self.trader = trader
        self.risk_manager = risk_manager
        self.active = config.sniper_active
        self.seen_listings = set()
        self.recent_detections = []
        self._load_seen()

    def _load_seen(self):
        try:
            with open('/data/seen_listings.json', 'r') as f:
                self.seen_listings = set(json.load(f))
        except Exception:
            self.seen_listings = set()

    def _save_seen(self):
        try:
            with open('/data/seen_listings.json', 'w') as f:
                json.dump(list(self.seen_listings), f)
        except Exception:
            pass

    def run(self):
        log.info("Listing sniper started.")
        while True:
            if self.active:
                try:
                    self._check_announcements()
                    self._check_new_pairs()
                except Exception as e:
                    log.error(f"Sniper check error: {e}")
            time.sleep(10)  # Check every 10 seconds

    def _check_announcements(self):
        """Scrape Binance announcement page for new listings"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; AutoTrader/1.0)',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            resp = requests.get(BINANCE_ANNOUNCEMENTS, headers=headers, timeout=10)
            if resp.status_code != 200:
                return

            soup = BeautifulSoup(resp.text, 'html.parser')
            articles = soup.find_all('a', href=True)

            for article in articles:
                title = article.get_text(strip=True)
                href = article['href']
                if 'listing' in title.lower() or 'will list' in title.lower():
                    listing_id = href.split('/')[-1] if href else title[:50]
                    if listing_id not in self.seen_listings:
                        self.seen_listings.add(listing_id)
                        self._save_seen()
                        symbol = self._extract_symbol(title)
                        if symbol:
                            log.info(f"New listing detected: {symbol} — {title}")
                            self._handle_new_listing(symbol, title, 'announcement')

        except Exception as e:
            log.debug(f"Announcement check failed: {e}")

    def _check_new_pairs(self):
        """Check Binance exchange info for brand new trading pairs"""
        try:
            info = self.trader.client.get_exchange_info()
            current_symbols = {s['symbol'] for s in info['symbols'] if 'USDT' in s['symbol']}

            if not hasattr(self, '_known_symbols'):
                self._known_symbols = current_symbols
                return

            new_symbols = current_symbols - self._known_symbols
            if new_symbols:
                for symbol in new_symbols:
                    log.info(f"New trading pair detected on Binance: {symbol}")
                    self._handle_new_listing(symbol, f"New pair listed: {symbol}", 'websocket')
                self._known_symbols = current_symbols

        except Exception as e:
            log.debug(f"Pair check failed: {e}")

    def _extract_symbol(self, title: str) -> str | None:
        """Extract coin symbol from announcement title"""
        import re
        # Look for patterns like (BTC), (ETH), (NEWCOIN)
        match = re.search(r'\(([A-Z]{2,10})\)', title)
        if match:
            symbol = match.group(1) + 'USDT'
            return symbol
        # Look for "will list X" patterns
        match = re.search(r'(?:list|listing)\s+([A-Z]{2,10})', title, re.IGNORECASE)
        if match:
            return match.group(1).upper() + 'USDT'
        return None

    def _handle_new_listing(self, symbol: str, title: str, source: str):
        """Process a detected new listing"""
        detection = {
            'symbol': symbol,
            'title': title,
            'source': source,
            'time': datetime.now().strftime('%H:%M:%S'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'status': 'detected',
            'ai_score': None,
            'action': None
        }

        # AI score the listing
        if self.config.ai_filter_enabled:
            score = self._ai_score_listing(symbol, title)
            detection['ai_score'] = score

            if score < self.config.ai_min_score:
                detection['status'] = 'filtered'
                detection['action'] = f'Skipped (AI score {score}% < {self.config.ai_min_score}%)'
                log.info(f"Sniper filtered {symbol} — AI score {score}% too low")
                self.recent_detections.insert(0, detection)
                return

        # Risk check
        approved, reason = self.risk_manager.check_snipe(symbol)
        if not approved:
            detection['status'] = 'blocked'
            detection['action'] = f'Risk blocked: {reason}'
            self.recent_detections.insert(0, detection)
            return

        # Approval mode — send Telegram alert and wait
        if self.config.approval_mode:
            self._send_telegram_alert(symbol, detection.get('ai_score', '?'), title)
            detection['status'] = 'pending_approval'
            detection['action'] = 'Telegram alert sent — awaiting approval'
            self.recent_detections.insert(0, detection)
            return

        # Auto mode — execute immediately
        self._execute_snipe(symbol, detection)

    def _execute_snipe(self, symbol: str, detection: dict):
        """Execute the snipe trade"""
        try:
            # Wait for trading to open
            self._wait_for_trading(symbol)

            result = self.trader.snipe_listing(symbol, self.config.sniper_budget_usdt)
            if result['success']:
                detection['status'] = 'bought'
                detection['action'] = f"Bought ${self.config.sniper_budget_usdt} worth"
                log.info(f"Snipe executed: {symbol}")

                # Set stop loss and take profit monitoring
                from threading import Thread
                monitor = Thread(
                    target=self._monitor_snipe_position,
                    args=(symbol, result),
                    daemon=True
                )
                monitor.start()
            else:
                detection['status'] = 'failed'
                detection['action'] = f"Trade failed: {result.get('error')}"

        except Exception as e:
            detection['status'] = 'error'
            detection['action'] = str(e)
            log.error(f"Snipe execution error for {symbol}: {e}")

        self.recent_detections.insert(0, detection)

    def _wait_for_trading(self, symbol: str, max_wait: int = 60):
        """Wait until the symbol is tradeable"""
        for _ in range(max_wait):
            try:
                self.trader.client.get_symbol_ticker(symbol=symbol)
                return
            except Exception:
                time.sleep(1)

    def _monitor_snipe_position(self, symbol: str, order: dict):
        """Monitor a sniped position for TP/SL"""
        buy_price = None
        tp_pct = self.config.sniper_tp_pct / 100
        sl_pct = self.config.sniper_sl_pct / 100
        max_hold_minutes = 30

        start = time.time()
        while time.time() - start < max_hold_minutes * 60:
            try:
                ticker = self.trader.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])

                if buy_price is None:
                    buy_price = current_price
                    log.info(f"Snipe {symbol} — buy price: ${buy_price:.6f}")
                    continue

                change = (current_price - buy_price) / buy_price
                log.debug(f"Snipe {symbol} — change: {change*100:.2f}%")

                if change >= tp_pct:
                    log.info(f"Snipe TP hit for {symbol}: +{change*100:.1f}%")
                    self.trader.execute_trade(f"{symbol.replace('USDT', '')}/USDT", 'sell', 100)
                    return

                if change <= -sl_pct:
                    log.info(f"Snipe SL hit for {symbol}: {change*100:.1f}%")
                    self.trader.execute_trade(f"{symbol.replace('USDT', '')}/USDT", 'sell', 100)
                    return

            except Exception as e:
                log.warning(f"Snipe monitor error: {e}")

            time.sleep(5)

        # Time limit reached — force sell
        log.info(f"Snipe time limit reached for {symbol} — force selling")
        try:
            self.trader.execute_trade(f"{symbol.replace('USDT', '')}/USDT", 'sell', 100)
        except Exception as e:
            log.error(f"Force sell failed: {e}")

    def _ai_score_listing(self, symbol: str, title: str) -> int:
        """Use Claude to score a new listing's potential"""
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 200,
                    "messages": [{
                        "role": "user",
                        "content": f"""Rate this new Binance listing from 0-100 based on sniping potential:
Symbol: {symbol}
Announcement: {title}

Consider: liquidity risk, pump potential, rug pull risk, project legitimacy signals.
Return ONLY a JSON: {{"score": 0-100, "reason": "brief reason"}}"""
                    }]
                },
                timeout=15
            )
            data = response.json()
            text = data['content'][0]['text']
            result = json.loads(text[text.find('{'):text.rfind('}')+1])
            return int(result.get('score', 50))
        except Exception:
            return 50  # Default neutral score

    def _send_telegram_alert(self, symbol: str, ai_score, title: str):
        """Send Telegram notification for approval"""
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            return
        msg = (
            f"🎯 *New Listing Detected!*\n\n"
            f"Symbol: `{symbol}`\n"
            f"AI Score: *{ai_score}/100*\n"
            f"Source: {title[:80]}\n\n"
            f"Max spend: ${self.config.sniper_budget_usdt}\n"
            f"TP: +{self.config.sniper_tp_pct}% | SL: -{self.config.sniper_sl_pct}%\n\n"
            f"Reply /buy_{symbol} to execute or /skip to ignore"
        )
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage",
                json={
                    'chat_id': self.config.telegram_chat_id,
                    'text': msg,
                    'parse_mode': 'Markdown'
                },
                timeout=5
            )
        except Exception as e:
            log.warning(f"Telegram alert failed: {e}")
