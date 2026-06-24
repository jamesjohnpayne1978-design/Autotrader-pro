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
import os
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
    # Where the persistent snipe history lives. Survives Railway redeploys.
    _HISTORY_PATH = '/data/sniper_history.json'
    # Max history entries to keep on disk
    _HISTORY_MAX = 100

    def __init__(self, config, trader, risk_manager):
        self.config = config
        self.trader = trader
        self.risk_manager = risk_manager
        self.active = config.sniper_active
        self.seen_symbols = set()
        self.seen_news_ids = set()
        self.recent_detections = []
        self.history = []
        self._load_seen()
        self._load_history()
        log.info(f"Sniper initialised - watching for new listings (history: {len(self.history)} entries)")

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

    # ============================================================
    # SNIPE HISTORY (persistent across restarts)
    # ============================================================

    def _load_history(self):
        """Load persistent snipe history from disk. Newest entries first."""
        try:
            if os.path.exists(self._HISTORY_PATH):
                with open(self._HISTORY_PATH, 'r') as f:
                    data = json.load(f) or []
                    self.history = data if isinstance(data, list) else []
        except Exception as e:
            log.warning(f"Could not load sniper history: {e}")
            self.history = []

    def _save_history(self):
        """Persist history list to disk."""
        try:
            os.makedirs(os.path.dirname(self._HISTORY_PATH), exist_ok=True)
            with open(self._HISTORY_PATH, 'w') as f:
                json.dump(self.history[:self._HISTORY_MAX], f, indent=2, default=str)
        except Exception as e:
            log.warning(f"Could not save sniper history: {e}")

    def _record_to_history(self, detection: dict, extra: dict = None):
        """Add a snipe to persistent history. Called on close, failure, or block.

        detection is the in-memory dict the sniper builds during execution.
        extra optionally overrides/adds fields (e.g. final exit price, pnl).
        """
        if not detection:
            return
        try:
            entry = {
                'symbol': detection.get('symbol'),
                'source': detection.get('source'),
                'status': detection.get('status'),
                'detected_date': detection.get('date'),
                'detected_time': detection.get('time'),
                'bought_at': detection.get('bought_at'),
                'entry_price': detection.get('buy_price'),
                'exit_price': detection.get('exit_price'),
                'qty': detection.get('qty'),
                'usdt_spent': detection.get('usdt_spent'),
                'result_pct': detection.get('change_pct'),
                'pnl_usdt': detection.get('exit_pnl_usdt'),
                'action': detection.get('action'),
                'tp_pct': detection.get('tp_pct'),
                'sl_pct': detection.get('sl_pct'),
                'oco_placed': detection.get('oco_placed'),
                'oco_tp_price': detection.get('oco_tp_price'),
                'oco_sl_price': detection.get('oco_sl_price'),
                'closed_at': datetime.now().isoformat(timespec='seconds'),
                'error': detection.get('oco_error'),
            }
            if extra:
                entry.update(extra)

            # Calculate hold duration in minutes if we have both timestamps
            try:
                if entry.get('bought_at') and entry.get('closed_at'):
                    bought = datetime.fromisoformat(entry['bought_at'])
                    closed = datetime.fromisoformat(entry['closed_at'])
                    entry['duration_minutes'] = round((closed - bought).total_seconds() / 60, 1)
            except Exception:
                pass

            # Insert newest-first, cap at MAX
            self.history.insert(0, entry)
            self.history = self.history[:self._HISTORY_MAX]
            self._save_history()
            log.info(f"Snipe recorded to history: {entry.get('symbol')} "
                     f"status={entry.get('status')} result={entry.get('result_pct')}%")
        except Exception as e:
            log.warning(f"Could not record snipe to history: {e}")

    def get_history(self, limit=50):
        """Returns the most recent snipes (used by /api/sniper/history)."""
        return self.history[:limit]

    def clear_history(self):
        """Wipe the persistent history (used by /api/sniper/history/clear)."""
        self.history = []
        self._save_history()
        return True

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
                    # Skip symbols already on the permanent block list (tokenized
                    # stocks, region-restricted products). Saves the API call and
                    # the misleading "NEW LISTING DETECTED" alert.
                    try:
                        if hasattr(self.trader, 'is_symbol_blocked') and self.trader.is_symbol_blocked(symbol):
                            log.info(f"Skipping {symbol} - on permanent block list")
                            continue
                    except Exception:
                        pass
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
                        # Block-list early-out (see exchange check for rationale)
                        try:
                            if hasattr(self.trader, 'is_symbol_blocked') and self.trader.is_symbol_blocked(symbol):
                                log.info(f"Skipping {symbol} from news - on permanent block list")
                                continue
                        except Exception:
                            pass
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
            self._record_to_history(detection)
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
        TP and SL at the exchange level. The OCO is the ONLY exit mechanism -
        there is no force-sell timer. Position runs until OCO triggers OR the
        user manually closes."""
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
                            f"_OCO is the only exit - position runs until TP or SL hits._"
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
                self._record_to_history(detection, extra={'error': str(err)[:200]})
        except Exception as e:
            detection['status'] = 'error'
            detection['action'] = str(e)
            log.error(f"Snipe error for {symbol}: {e}")
            self._send_telegram_failed(symbol, str(e))
            self._record_to_history(detection, extra={'error': str(e)[:200]})

    def _get_actual_sell_fill(self, symbol: str, since_ms: int):
        """Query Binance for the actual sell trade(s) that closed our position.

        Returns (fill_price, fill_qty, was_oco) or (None, None, False) if no
        sell found. Used to report ACTUAL fill prices instead of stale ticker
        prices when announcing snipe closes.

        was_oco is True if the sell came from an OCO order (orderListId != -1),
        which tells us whether TP or SL leg fired vs. a manual sell.
        """
        try:
            trades = self.trader.client.get_my_trades(symbol=symbol, limit=20) or []
            # Filter to sells AFTER our buy, take volume-weighted average price
            sells = [t for t in trades
                     if not t['isBuyer'] and int(t['time']) >= since_ms]
            if not sells:
                return None, None, False
            total_qty = sum(float(t['qty']) for t in sells)
            total_val = sum(float(t['price']) * float(t['qty']) for t in sells)
            avg_price = total_val / total_qty if total_qty > 0 else 0
            # If ANY sell has orderListId != -1, the close came from the OCO
            was_oco = any(int(t.get('orderListId', -1)) != -1 for t in sells)
            return avg_price, total_qty, was_oco
        except Exception as e:
            log.debug(f"Could not fetch sell fills for {symbol}: {e}")
            return None, None, False

    def _monitor_position(self, symbol: str, detection: dict = None):
        """Watches a sniped position for dashboard purposes - updates current
        price and P&L in the detection dict every few seconds. The actual exit
        (TP/SL) is handled by the OCO order at Binance, not by this thread.

        When the position closes, queries Binance for the ACTUAL sell fill
        price (not the live ticker) so the result message is accurate.

        Stops when the position is sold (balance goes to ~0) or after 24 hours
        as a safety bound. No force-sell."""
        buy_price = detection.get('buy_price') if detection else None
        tp_pct = (detection.get('tp_pct') if detection else None) or float(self.config.sniper_tp_pct)
        sl_pct = (detection.get('sl_pct') if detection else None) or float(self.config.sniper_sl_pct)
        # Buy timestamp in ms - used to find sell trades that came after the buy
        bought_at_ms = int(time.time() * 1000)
        try:
            if detection and detection.get('bought_at'):
                bought_at_ms = int(datetime.fromisoformat(detection['bought_at']).timestamp() * 1000)
        except Exception:
            pass

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
                        # Position closed - find the ACTUAL fill price from
                        # Binance trade history, not the live ticker. This
                        # matters because volatile new listings can bounce
                        # several % between the SL firing and the next monitor
                        # cycle 15s later.
                        actual_price, actual_qty, was_oco = self._get_actual_sell_fill(symbol, bought_at_ms)

                        if actual_price and actual_price > 0:
                            actual_change_pct = ((actual_price - buy_price) / buy_price) * 100
                            pnl_usdt = round((actual_price - buy_price) * (actual_qty or original_qty), 2)
                            # Classify exit by ACTUAL fill, not by current ticker
                            if was_oco:
                                if actual_change_pct > 0:
                                    exit_type = '🎯 TP HIT'
                                    status = 'tp_hit'
                                else:
                                    exit_type = '🛑 SL HIT'
                                    status = 'sl_hit'
                            else:
                                exit_type = '✅ Closed'
                                status = 'closed'

                            if detection is not None:
                                detection['status'] = status
                                detection['exit_price'] = actual_price
                                detection['exit_pnl_usdt'] = pnl_usdt
                                detection['action'] = f"{exit_type} {actual_change_pct:+.1f}%"
                                detection['monitoring'] = False

                            sign = '+' if pnl_usdt >= 0 else '-'
                            self._telegram_post(
                                f"{exit_type} *Snipe closed*\n"
                                f"`{symbol}`\n"
                                f"Entry: ${buy_price:.6f}\n"
                                f"Exit:  ${actual_price:.6f}\n"
                                f"Result: {actual_change_pct:+.1f}% ({sign}${abs(pnl_usdt):.2f})"
                            )
                            log.info(f"Snipe position closed: {symbol} actual fill {actual_change_pct:+.1f}% "
                                     f"(ticker was {change_pct:+.1f}%) via {exit_type}")
                            # Persist this completed snipe to history
                            self._record_to_history(detection, extra={
                                'exit_type': exit_type,
                                'result_pct': round(actual_change_pct, 2),
                            })
                        else:
                            # Fallback - couldn't fetch actual fill, use ticker
                            # but flag the uncertainty so user knows
                            if detection is not None:
                                detection['status'] = 'closed'
                                detection['action'] = f"Closed {change_pct:+.1f}% (approx)"
                                detection['monitoring'] = False
                            self._telegram_post(
                                f"✅ *Snipe closed*\n`{symbol}`\n"
                                f"Result: ~{change_pct:+.1f}% (ticker estimate - check trade history for exact)"
                            )
                            log.info(f"Snipe position closed: {symbol} ~{change_pct:+.1f}% (ticker only)")
                            self._record_to_history(detection, extra={
                                'exit_type': '✅ Closed (ticker estimate)',
                                'result_pct': round(change_pct, 2),
                            })
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
        """Alert: sniper detected a listing and is attempting to buy.

        Note: removed the misleading 'Max hold: 30 min' line - there's no
        30-min force-sell. The OCO at Binance is the only exit mechanism.
        """
        msg = (
            f"🎯 *NEW LISTING DETECTED*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Source: {title[:80]}\n\n"
            f"Attempting buy: ${budget:.0f}\n"
            f"TP: +{self.config.sniper_tp_pct}% · SL: -{self.config.sniper_sl_pct}%\n"
            f"Exit: OCO at Binance (no time limit)"
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
