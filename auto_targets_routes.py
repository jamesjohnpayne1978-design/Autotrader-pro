"""
auto_targets_routes.py - Flask routes for the ranker-driven concentration targets.

Kept in its own module so app.py needs only a tiny hook (see register call).

Routes
------
GET  /api/concentration/auto-targets
        Phone-friendly status page (shows on/off, current picks, scores,
        challenger progress, recent changes) with a button to flip the mode.
        Add ?json=1 for the raw JSON instead.
POST /api/concentration/auto-targets
        Flip the mode. Form button posts here; JSON {"enabled": true|false}
        sets it explicitly. Redirects back to the page (or returns JSON if
        the caller asked for JSON).

Changing the mode is a POST on purpose: a link preview or prefetch of a GET
URL must never be able to change how real money is allocated.
"""

import logging
from html import escape

from flask import jsonify, request, redirect

log = logging.getLogger(__name__)

_ROUTE = '/api/concentration/auto-targets'


def _wants_json():
    if request.args.get('json'):
        return True
    accept = request.headers.get('Accept', '')
    return 'application/json' in accept and 'text/html' not in accept


def _fmt(v, suffix=''):
    if v is None:
        return '-'
    return f"{v}{suffix}"


def _render_page(alloc):
    enabled = alloc.auto_enabled()
    conc_on = alloc.is_enabled()
    color = '#00d4a0' if enabled else '#94a3b8'
    state_text = 'ON' if enabled else 'OFF'

    warn = ''
    if not conc_on:
        warn = ('<div style="margin-top:16px;padding:12px;background:#3a2a10;color:#fbbf24;'
                'border-radius:10px;font-size:0.85rem;">Concentration mode is OFF, so nothing '
                'trades yet. Auto targets only act while concentration mode is on.</div>')

    body = ''
    if enabled:
        try:
            a = alloc._auto_status()
        except Exception as e:  # never break the page
            a = None
            body = f'<div style="color:#f87171;">Status unavailable: {escape(str(e))}</div>'
        if a:
            rows = [
                f'<div style="display:flex;justify-content:space-between;padding:8px 0;'
                f'border-bottom:1px solid #1f2a3d;"><span>BTC (anchor)</span>'
                f'<b>{a["anchor_weight_pct"]}%</b></div>'
            ]
            for s in a['slots']:
                name = escape(s['asset'] or 'cash (empty slot)')
                detail = ''
                if s['asset']:
                    detail = (f'<div style="color:#94a3b8;font-size:0.78rem;">score {_fmt(s["score"])} '
                              f'· rank {_fmt(s["rank"])} · RSI {_fmt(s["rsi_14d"])} · held {s["held_hours"]}h'
                              f'{" · min-hold left " + str(s["min_hold_remaining_hours"]) + "h" if s["min_hold_remaining_hours"] else ""}</div>')
                rows.append(
                    f'<div style="padding:8px 0;border-bottom:1px solid #1f2a3d;">'
                    f'<div style="display:flex;justify-content:space-between;"><span>{name}</span>'
                    f'<b>{s["weight_pct"]}%</b></div>{detail}</div>')
            streaks = a.get('challenger_streaks') or {}
            if streaks:
                txt = ', '.join(f'{escape(k.replace("USDT", ""))}: {v}/{a["rules"]["confirm_rankings"]}'
                                for k, v in streaks.items())
                rows.append(f'<div style="padding:8px 0;color:#94a3b8;font-size:0.85rem;">'
                            f'Challenger building: {txt}</div>')
            stale = ' (STALE - picks frozen)' if a.get('ranking_stale') else ''
            rows.append(f'<div style="padding:8px 0;color:#94a3b8;font-size:0.8rem;">'
                        f'Ranker data: {_fmt(a.get("ranking_age_hours"), "h old")}{stale}</div>')
            recent = a.get('recent_changes') or []
            if recent:
                items = ''.join(
                    f'<div style="font-size:0.8rem;color:#94a3b8;padding:2px 0;">'
                    f'{escape(str(c.get("at", ""))[:16])} · {escape(c.get("type", ""))}: '
                    f'{escape((c.get("out") or "-").replace("USDT", ""))} → '
                    f'{escape((c.get("in") or "cash").replace("USDT", ""))}</div>'
                    for c in reversed(recent))
                rows.append(f'<div style="padding-top:10px;"><div style="font-size:0.75rem;'
                            f'text-transform:uppercase;color:#94a3b8;">Recent changes</div>{items}</div>')
            body = ('<div style="margin-top:20px;padding:16px;background:#1a2233;border-radius:12px;'
                    'text-align:left;font-size:0.95rem;">' + ''.join(rows) + '</div>')

    button_label = 'Turn OFF' if enabled else 'Turn ON'
    explain = ('BTC stays a permanent 20% anchor. The 40% and 25% slots follow the momentum '
               'ranker, with confirmation, minimum-hold and anti-froth rules.')

    return f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Auto Targets</title>
</head>
<body style="margin:0;padding:32px 20px;background:#0d1421;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,sans-serif;min-height:100vh;">
<div style="max-width:420px;margin:0 auto;text-align:center;">
    <div style="font-size:0.85rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Auto Targets</div>
    <div style="font-size:3rem;font-weight:800;color:{color};margin:12px 0;">{state_text}</div>
    <div style="font-size:0.9rem;color:#94a3b8;line-height:1.5;">{explain}</div>
    {warn}
    {body}
    <form method="post" action="{_ROUTE}" style="margin-top:28px;">
        <button type="submit" style="width:100%;padding:16px;background:#1d4ed8;color:white;border:0;border-radius:12px;font-weight:600;font-size:1rem;">{button_label}</button>
    </form>
    <div style="margin-top:14px;display:flex;flex-direction:column;gap:8px;">
        <a href="{_ROUTE}?json=1" style="display:block;padding:12px;color:#94a3b8;text-decoration:none;font-size:0.85rem;">View raw status (JSON)</a>
        <a href="/" style="display:block;padding:12px;color:#94a3b8;text-decoration:none;font-size:0.85rem;">Back to dashboard</a>
    </div>
</div>
</body>
</html>"""


def register_auto_targets_routes(app, get_allocator):
    """Attach the routes. get_allocator is a zero-arg callable returning the
    live WinnersAllocator (or None before the trader has initialised)."""

    @app.route(_ROUTE, methods=['GET', 'POST'])
    def auto_targets():
        alloc = get_allocator()
        if alloc is None:
            msg = 'Allocator not initialised yet - try again in a minute'
            if request.method == 'POST' or _wants_json():
                return jsonify({'error': msg}), 400
            return f'<p style="font-family:sans-serif;padding:24px;">{msg}</p>', 503

        if request.method == 'POST':
            payload = request.get_json(silent=True) or {}
            if 'enabled' in payload:
                new_state = bool(payload['enabled'])
            else:
                new_state = not alloc.auto_enabled()
            alloc.set_auto_enabled(new_state)
            log.info(f"Auto targets toggled via API: {new_state}")
            if request.is_json or _wants_json():
                return jsonify({'success': True, 'auto_targets_enabled': new_state})
            return redirect(_ROUTE, code=303)

        if _wants_json():
            return jsonify({
                'auto_targets_enabled': alloc.auto_enabled(),
                'concentration_mode_enabled': alloc.is_enabled(),
                'auto': alloc._auto_status() if alloc.auto_enabled() else None,
            })
        return _render_page(alloc)
