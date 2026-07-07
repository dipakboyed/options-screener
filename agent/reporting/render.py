from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _fmt_money(val: Optional[float], prefix: str = "$") -> str:
    if val is None:
        return "—"
    return f"{prefix}{val:,.2f}"


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}%"


def _render_cc_recommendations(
    recommendations: List[Dict[str, Any]],
    html_parts: List[str],
) -> None:
    """Render the Covered Call recommendation table."""
    if not recommendations:
        return

    yes_count = sum(1 for r in recommendations if r["recommend"] == "Yes")
    borderline_count = sum(1 for r in recommendations if r["recommend"] == "Borderline")
    total = len(recommendations)

    html_parts.append("<details open class='section section-cc-rec'>")
    html_parts.append(
        f"<summary>Covered Call Recommendations"
        f"<span class='count'>"
        f"{yes_count} Yes &nbsp;·&nbsp; {borderline_count} Borderline &nbsp;·&nbsp; {total} total"
        f"</span></summary>"
    )
    html_parts.append("<div class='cc-rec-body'>")
    html_parts.append("<table class='cc-rec-table'>")
    html_parts.append(
        "<thead><tr>"
        "<th>Ticker</th>"
        "<th>Term</th>"
        "<th>Ann. Yield</th>"
        "<th>Recommend</th>"
        "<th>Current Price</th>"
        "<th>Strike</th>"
        "<th>% OTM</th>"
        "<th>Expiration</th>"
        "<th>DTE</th>"
        "<th>Premium</th>"
        "<th>Delta</th>"
        "<th>IVR *</th>"
        "<th>Max Profit</th>"
        "<th>Breakeven</th>"
        "<th>Flags</th>"
        "<th>Why</th>"
        "</tr></thead><tbody>"
    )

    verdict_class = {"Yes": "rec-yes", "No": "rec-no", "Borderline": "rec-borderline"}

    _GRP_COLORS = ("#ffffff", "#e8f0ff")
    prev_group = None
    group_idx = -1
    for rec in recommendations:
        group = (rec["ticker"], rec.get("term", ""))
        if group != prev_group:
            group_idx += 1
            prev_group = group
        row_bg = _GRP_COLORS[group_idx % 2]
        ticker = rec["ticker"]
        verdict = rec["recommend"]
        fidelity_url = f"https://digital.fidelity.com/ftgw/digital/options-research/?symbol={ticker}"

        spot_val = rec.get("spot")
        strike_val = rec.get("strike")
        ann_yield = rec.get("annualized_yield")
        yield_display = f"{ann_yield:.1%}" if ann_yield is not None else "—"

        otm_pct = (
            f"{(strike_val - spot_val) / spot_val * 100:.1f}%"
            if spot_val and strike_val else "—"
        )

        ivr_display = _fmt_pct(rec.get("ivr"))
        if "proxy" in (rec.get("ivr_source") or ""):
            ivr_display += " ★"

        delta_raw = rec.get("delta")
        delta_display = f"{abs(float(delta_raw)):.3f}" if delta_raw is not None else "—"

        # Build flags cell
        flags: List[str] = []
        if rec.get("near_resistance"):
            flags.append("<span class='flag-res' title='Strike near resistance level'>&#9650; resistance</span>")
        if rec.get("near_round_number"):
            flags.append("<span class='flag-round' title='Strike near $5 round number'>&#9675; round#</span>")
        if rec.get("below_min_price"):
            min_p = rec.get("min_acceptable_price")
            min_str = f"${min_p:.2f}" if min_p is not None else "min"
            flags.append(f"<span class='flag-below' title='Strike below your minimum acceptable price'>&#9888; below {min_str}</span>")
        flags_html = " ".join(flags) if flags else "—"

        css = verdict_class.get(verdict, "")
        html_parts.append(
            f"<tr style='background-color:{row_bg}'>"
            f"<td><a href='{escape(fidelity_url)}' target='_blank' rel='noopener noreferrer'><strong>{escape(ticker)}</strong></a></td>"
            f"<td><strong>{escape(rec.get('term', ''))}</strong></td>"
            f"<td>{escape(yield_display)}</td>"
            f"<td class='{css}'><strong>{escape(verdict)}</strong></td>"
            f"<td>{_fmt_money(spot_val)}</td>"
            f"<td>{_fmt_money(strike_val)}</td>"
            f"<td>{escape(otm_pct)}</td>"
            f"<td>{escape(str(rec.get('expiration') or '—'))}</td>"
            f"<td>{rec.get('dte') or '—'}</td>"
            f"<td>{_fmt_money(rec.get('premium'))}</td>"
            f"<td>{delta_display}</td>"
            f"<td>{escape(ivr_display)}</td>"
            f"<td>{_fmt_money(rec.get('max_profit'))}</td>"
            f"<td>{_fmt_money(rec.get('downside_breakeven'))}</td>"
            f"<td>{flags_html}</td>"
            f"<td class='reason-cell'>{escape(rec.get('reason', ''))}</td>"
            f"</tr>"
        )

    html_parts.append("</tbody></table>")

    has_proxy = any("proxy" in (r.get("ivr_source") or "") for r in recommendations)
    if has_proxy:
        html_parts.append(
            "<p class='rec-footnote'>★ IVR shown for reference only — it does <strong>not</strong> affect the "
            "recommendation verdict. Computed as HV Rank (current 20-day HV vs 1-year HV range). "
            "True IV Rank requires historical implied volatility data.</p>"
        )

    html_parts.append(
        "<div class='exit-rules'>"
        "<strong>Exit Rules (Covered Calls):</strong>"
        "<ul>"
        "<li>Close the position at <strong>70% of max profit</strong> — capture most of the premium early.</li>"
        "<li>Close or roll up/out if the stock rallies and unrealised loss reaches <strong>2&times; the premium received</strong>.</li>"
        "<li>Roll to a <strong>higher strike or later expiration</strong> if assignment is imminent and you want to retain the shares.</li>"
        "</ul>"
        "</div>"
    )

    html_parts.append("</div>")  # cc-rec-body
    html_parts.append("</details>")


def _render_csp_recommendations(
    recommendations: List[Dict[str, Any]],
    html_parts: List[str],
) -> None:
    """Render the CSP recommendation table at the top of the report."""
    if not recommendations:
        return

    yes_count = sum(1 for r in recommendations if r["recommend"] == "Yes")
    borderline_count = sum(1 for r in recommendations if r["recommend"] == "Borderline")
    total = len(recommendations)

    html_parts.append("<details open class='section section-rec'>")
    html_parts.append(
        f"<summary>CSP Recommendations"
        f"<span class='count'>"
        f"{yes_count} Yes &nbsp;·&nbsp; {borderline_count} Borderline &nbsp;·&nbsp; {total} total"
        f"</span></summary>"
    )

    html_parts.append("<div class='rec-body'>")
    html_parts.append("<table class='rec-table'>")
    html_parts.append(
        "<thead><tr>"
        "<th>Ticker</th>"
        "<th>Term</th>"
        "<th>Ann. Yield</th>"
        "<th>Recommend</th>"
        "<th>Current Price</th>"
        "<th>Strike</th>"
        "<th>% to Strike</th>"
        "<th>Expiration</th>"
        "<th>DTE</th>"
        "<th>Premium</th>"
        "<th>Delta</th>"
        "<th>IVR *</th>"
        "<th>Max Profit</th>"
        "<th>Breakeven</th>"
        "<th>Cash Req.</th>"
        "<th>Why</th>"
        "</tr></thead><tbody>"
    )

    verdict_class = {"Yes": "rec-yes", "No": "rec-no", "Borderline": "rec-borderline"}

    _GRP_COLORS = ("#ffffff", "#e8f0ff")
    prev_term = None
    group_idx = -1
    for rec in recommendations:
        term = rec.get("term", "")
        if term != prev_term:
            group_idx += 1
            prev_term = term
        row_bg = _GRP_COLORS[group_idx % 2]
        ticker = rec["ticker"]
        verdict = rec["recommend"]
        fidelity_url = (
            f"https://digital.fidelity.com/ftgw/digital/options-research/?symbol={ticker}"
        )
        ivr_display = _fmt_pct(rec.get("ivr"))
        if rec.get("ivr_source"):
            ivr_display += " ★" if "proxy" in (rec.get("ivr_source") or "") else ""

        ann_yield = rec.get("annualized_yield")
        yield_display = f"{ann_yield:.1%}" if ann_yield is not None else "—"

        near_flags = []
        if rec.get("near_support"):
            near_flags.append("✓ support")
        if rec.get("near_round_number"):
            near_flags.append("○ round#")
        near_str = " ".join(near_flags)
        reason_full = rec.get("reason", "")
        if near_str:
            reason_full = f"{reason_full} [{near_str}]" if reason_full else near_str

        css = verdict_class.get(verdict, "")
        delta_raw = rec.get("delta")
        delta_display = f"{abs(float(delta_raw)):.3f}" if delta_raw is not None else "—"
        spot_val = rec.get("spot")
        strike_val = rec.get("strike")
        pct_to_strike = (
            f"{(strike_val - spot_val) / spot_val * 100:.1f}%"
            if spot_val and strike_val
            else "—"
        )
        html_parts.append(
            f"<tr style='background-color:{row_bg}'>"
            f"<td><a href='{escape(fidelity_url)}' target='_blank' rel='noopener noreferrer'><strong>{escape(ticker)}</strong></a></td>"
            f"<td><strong>{escape(rec.get('term', ''))}</strong></td>"
            f"<td>{escape(yield_display)}</td>"
            f"<td class='{css}'><strong>{escape(verdict)}</strong></td>"
            f"<td>{_fmt_money(spot_val)}</td>"
            f"<td>{_fmt_money(strike_val)}</td>"
            f"<td>{escape(pct_to_strike)}</td>"
            f"<td>{escape(str(rec.get('expiration') or '—'))}</td>"
            f"<td>{rec.get('dte') or '—'}</td>"
            f"<td>{_fmt_money(rec.get('premium'))}</td>"
            f"<td>{delta_display}</td>"
            f"<td>{escape(ivr_display)}</td>"
            f"<td>{_fmt_money(rec.get('max_profit'))}</td>"
            f"<td>{_fmt_money(rec.get('breakeven'))}</td>"
            f"<td>{_fmt_money(rec.get('cash_required'))}</td>"
            f"<td class='reason-cell'>{escape(reason_full)}</td>"
            f"</tr>"
        )

    html_parts.append("</tbody></table>")

    # IVR footnote
    has_proxy = any("proxy" in (r.get("ivr_source") or "") for r in recommendations)
    if has_proxy:
        html_parts.append(
            "<p class='rec-footnote'>★ IVR is a proxy calculated from the option IV (or current 20-day HV) "
            "relative to the historical HV range over the available price history. "
            "True IV Rank requires historical implied volatility data not available via yfinance.</p>"
        )

    # Exit rules
    html_parts.append(
        "<div class='exit-rules'>"
        "<strong>Exit Rules:</strong>"
        "<ul>"
        "<li>Close the position at <strong>50–70% of max profit</strong> — lock in gains early.</li>"
        "<li>Close or roll if unrealised loss reaches <strong>2× the premium received</strong>.</li>"
        "</ul>"
        "</div>"
    )

    html_parts.append("</div>")  # rec-body
    html_parts.append("</details>")


def _write_recommendations_csv(
    csv_path: Path,
    cc_recommendations: List[Dict[str, Any]],
    csp_recommendations: List[Dict[str, Any]],
) -> None:
    """Write the simplified recommendation rows (CC + CSP) to CSV.

    Replaces the previous full candidate dump. Both strategies share one schema
    so the file stays compact and matches what the HTML report shows.
    """
    columns = [
        "type", "ticker", "term", "recommend", "expiration", "dte", "spot",
        "strike", "premium", "annualized_yield", "delta", "ivr", "max_profit",
        "breakeven", "reason",
    ]

    rows: List[Dict[str, Any]] = []
    for rec in cc_recommendations:
        rows.append({
            "type": "CC",
            "ticker": rec.get("ticker"),
            "term": rec.get("term"),
            "recommend": rec.get("recommend"),
            "expiration": rec.get("expiration"),
            "dte": rec.get("dte"),
            "spot": rec.get("spot"),
            "strike": rec.get("strike"),
            "premium": rec.get("premium"),
            "annualized_yield": rec.get("annualized_yield"),
            "delta": rec.get("delta"),
            "ivr": rec.get("ivr"),
            "max_profit": rec.get("max_profit"),
            "breakeven": rec.get("downside_breakeven"),
            "reason": rec.get("reason"),
        })
    for rec in csp_recommendations:
        rows.append({
            "type": "CSP",
            "ticker": rec.get("ticker"),
            "term": rec.get("term"),
            "recommend": rec.get("recommend"),
            "expiration": rec.get("expiration"),
            "dte": rec.get("dte"),
            "spot": rec.get("spot"),
            "strike": rec.get("strike"),
            "premium": rec.get("premium"),
            "annualized_yield": rec.get("annualized_yield"),
            "delta": rec.get("delta"),
            "ivr": rec.get("ivr"),
            "max_profit": rec.get("max_profit"),
            "breakeven": rec.get("breakeven"),
            "reason": rec.get("reason"),
        })

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(csv_path, index=False)


def write_reports(
    candidates: List[Dict[str, Any]],
    config: Dict[str, Any],
    disclaimer: str,
    csp_recommendations: Optional[List[Dict[str, Any]]] = None,
    cc_recommendations: Optional[List[Dict[str, Any]]] = None,
    fallback_events: Optional[List[str]] = None,
) -> Tuple[str, str]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_day = date.today().isoformat()
    csv_path = output_dir / f"{run_day}_options_report.csv"
    html_path = output_dir / f"{run_day}_options_report.html"

    # CSV now exports the recommendation rows (CC + CSP), not the full candidate dump.
    _write_recommendations_csv(csv_path, cc_recommendations or [], csp_recommendations or [])

    html_parts: List[str] = []
    html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html_parts.append("<title>Options Report</title>")
    html_parts.append(
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#fff;}"
        "h1{margin-bottom:8px;}"
        "details{margin-bottom:6px;}"
        "summary{cursor:pointer;padding:7px 12px;border-radius:4px;user-select:none;list-style:none;display:flex;align-items:center;gap:6px;}"
        "summary::-webkit-details-marker{display:none;}"
        "summary::before{content:'▶';font-size:10px;transition:transform 0.15s;}"
        "details[open]>summary::before{transform:rotate(90deg);}"
        ".section>summary{font-size:16px;font-weight:700;background:#0969da;color:#fff;border:none;}"
        ".section>summary:hover{background:#0860ca;}"
        ".section>summary::before{color:#fff;}"
        "table{border-collapse:collapse;width:auto;margin:6px 0 6px 20px;}"
        "th,td{border:1px solid #d0d7de;padding:3px 6px;font-size:12px;text-align:left;white-space:nowrap;vertical-align:top;}"
        "th{background:#f6f8fa;font-weight:600;}"
        "tr:nth-child(even){background:#f9f9f9;}"
        ".count{font-weight:400;font-size:13px;opacity:0.85;margin-left:6px;}"
        ".note{font-size:12px;color:#444;padding:8px;background:#fff8c5;border:1px solid #e3b341;border-radius:4px;margin-bottom:12px;}"
        "a{color:inherit;}"
        ".section-rec>summary{background:#9a6700;color:#fff;font-size:17px;}"
        ".section-rec>summary:hover{background:#875d00;}"
        ".section-rec>summary::before{color:#fff;}"
        ".rec-body{padding:8px 12px 12px;}"
        ".rec-table{margin:0 0 10px 0;width:100%;}"
        ".rec-table th{background:#fdf0d5;font-size:12px;}"
        ".rec-table td{font-size:12px;}"
        ".rec-yes{background:#d4edda;color:#155724;font-weight:600;}"
        ".rec-no{background:#f8d7da;color:#721c24;font-weight:600;}"
        ".rec-borderline{background:#fff3cd;color:#856404;font-weight:600;}"
        ".reason-cell{white-space:normal;min-width:180px;max-width:320px;font-size:11px;color:#444;}"
        ".rec-footnote{font-size:11px;color:#666;margin:4px 0 8px;font-style:italic;}"
        ".exit-rules{font-size:12px;background:#f0f7ff;border:1px solid #b6d4fe;border-radius:4px;padding:8px 12px;margin-top:4px;}"
        ".exit-rules ul{margin:4px 0 0 16px;padding:0;}"
        ".exit-rules li{margin-bottom:2px;}"
        ".warn-banner{background:#fff3cd;border:2px solid #ffc107;border-radius:6px;padding:10px 14px;margin-bottom:14px;}"
        ".warn-banner h3{margin:0 0 6px;color:#856404;font-size:14px;}"
        ".warn-banner ul{margin:4px 0 0 18px;padding:0;color:#6c4a00;font-size:13px;}"
        ".warn-banner li{margin-bottom:3px;}"
        ".section-cc-rec>summary{background:#0d6efd;color:#fff;font-size:17px;}"
        ".section-cc-rec>summary:hover{background:#0b5ed7;}"
        ".section-cc-rec>summary::before{color:#fff;}"
        ".cc-rec-body{padding:8px 12px 12px;}"
        ".cc-rec-table{margin:0 0 10px 0;width:100%;}"
        ".cc-rec-table th{background:#dbeafe;font-size:12px;}"
        ".cc-rec-table td{font-size:12px;}"
        ".flag-res{color:#0d6efd;font-weight:600;}"
        ".flag-round{color:#6c757d;}"
        ".flag-below{color:#dc3545;font-weight:600;}"
        "</style></head><body>"
    )
    html_parts.append(f"<h1>Daily Options Screening Report &mdash; {run_day}</h1>")
    html_parts.append(f"<p class='note'>{escape(disclaimer)}</p>")

    # ── Provider fallback warnings ─────────────────────────────────────────────
    if fallback_events:
        # Deduplicate and extract affected tickers (first token before ':' or ' ')
        affected_tickers = sorted(set(e.split(":")[0].split(" ")[0] for e in fallback_events))
        tickers_str = ", ".join(affected_tickers) if affected_tickers else "some tickers"
        html_parts.append("<div class='warn-banner'>")
        html_parts.append(
            f"<h3>&#9888; Data Provider Warning: Public provider was inaccessible &mdash; "
            f"yfinance used as fallback for: {escape(tickers_str)}</h3>"
            f"<p style='margin:0;font-size:12px;color:#6c4a00;'>Check the run log for details.</p>"
        )
        html_parts.append("</div>")

    # ── Covered Call Recommendations (near-term headline) ──────────────────────
    if cc_recommendations:
        _render_cc_recommendations(cc_recommendations, html_parts)
    else:
        html_parts.append(
            "<p class='note'>No qualifying near-term covered calls today.</p>"
        )

    # ── CSP Recommendations (single best per ticker) ───────────────────────────
    if csp_recommendations:
        _render_csp_recommendations(csp_recommendations, html_parts)

    # Detailed per-ticker screening sections removed — report shows best recommendations only.

    html_parts.append("<hr>")
    html_parts.append(
        "<p><strong>Risk reminders:</strong> Assignment risk, overnight gaps, earnings/event shocks, "
        "liquidity deterioration, and tail-risk moves can cause losses.</p>"
    )
    html_parts.append("</body></html>")

    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    return str(csv_path), str(html_path)
