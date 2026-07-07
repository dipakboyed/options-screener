"""
Covered Call (CC) Recommendation Engine.

Analyses CALL candidates per ticker and produces multiple actionable
suggestions per term per ticker:
  - Short-Term:  DTE ≤ 14
  - Medium-Term: 15 ≤ DTE ≤ 28
  - Long-Term:   DTE > 28

At least one suggestion is always produced per term (even when no candidate
perfectly meets the delta range criteria).

Recommendation verdict per suggestion:
  - Yes:        |delta| in [delta_min, delta_max], no earnings conflict, strike ≥ min_acceptable_price
  - Borderline: delta outside target range, earnings too close, or strike below min_acceptable_price
  (IVR is shown for reference but does NOT affect the verdict.)

A strike near a resistance level is flagged as a positive signal (stock may
stall there, reducing assignment risk).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agent.recommendation.csp_recommender import compute_ivr_proxy


DEFAULT_CC_REC_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "max_recommendations": 50,
    "max_suggestions_per_term": 3,
    "earnings_buffer_days": 7,
    "delta_min": 0.10,
    "delta_max": 0.25,
    "use_resistance_filter": True,
    "resistance_pct_buffer": 0.02,
    "best_only": True,
    "max_per_term": 1,
    "keep_medium_long": True,
    "near_term_buckets": [
        {"label": "~1 Day", "min_dte": 0, "max_dte": 1},
        {"label": "~3 Day", "min_dte": 2, "max_dte": 3},
        {"label": "~5 Day", "min_dte": 4, "max_dte": 6},
    ],
    "min_acceptable_sale_prices": {},  # ticker -> float, optional per-ticker floor
}


# Fixed term buckets appended after the near-term buckets when keep_medium_long is on.
_MEDIUM_LONG_BUCKETS = [
    {"label": "Medium-Term", "min_dte": 15, "max_dte": 28},
    {"label": "Long-Term", "min_dte": 29, "max_dte": 10_000},
]


# ── Resistance levels ───────────────────────────────────────────────────────

def get_resistance_levels(price_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Return period high and 20-day swing high from available price history."""
    if price_df is None or price_df.empty:
        return {"high_52w": None, "swing_high_20d": None}

    high_col = (
        price_df["High"].astype(float)
        if "High" in price_df.columns
        else price_df["Close"].astype(float)
    )
    high_52w = float(high_col.max())
    swing_high_20d = float(high_col.rolling(20).max().iloc[-1]) if len(high_col) >= 20 else None
    return {"high_52w": high_52w, "swing_high_20d": swing_high_20d}


def _near_round_number(strike: float) -> bool:
    """True if strike is within 1% of the nearest $5 increment."""
    nearest = round(strike / 5) * 5
    return abs(strike - nearest) / max(strike, 1e-6) < 0.01


def _near_resistance(strike: float, resistance: Dict[str, Optional[float]], buffer: float) -> bool:
    """True if strike is within buffer% of a resistance level (stock may stall there)."""
    for level in (resistance.get("high_52w"), resistance.get("swing_high_20d")):
        if level is not None and level * (1 - buffer) <= strike <= level * (1 + buffer):
            return True
    return False


# ── Per-bucket recommendation ───────────────────────────────────────────────

def _make_base_row(
    ticker: str,
    term_label: str,
    spot: float,
    min_acceptable_price: Optional[float],
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "term": term_label,
        "recommend": "No",
        "reason": "",
        "spot": spot if spot > 0 else None,
        "strike": None,
        "expiration": None,
        "premium": None,
        "delta": None,
        "ivr": None,
        "ivr_source": None,
        "max_profit": None,
        "downside_breakeven": None,
        "near_resistance": False,
        "near_round_number": False,
        "below_min_price": False,
        "dte": None,
        "annualized_yield": None,
        "min_acceptable_price": min_acceptable_price,
    }


def _recommend_for_bucket(
    ticker: str,
    term_label: str,
    call_candidates: List[Dict[str, Any]],
    resistance: Dict[str, Optional[float]],
    technicals: Dict[str, float],
    earnings_date: Optional[date],
    min_acceptable_price: Optional[float],
    price_df: pd.DataFrame,
    rec_config: Dict[str, Any],
    max_suggestions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Return up to max_suggestions ranked suggestions for one term bucket.
    Always returns at least one row — even if no candidate perfectly meets
    the delta range, the best available option is shown as Borderline.
    IVR is computed for display only and does not affect the verdict.
    """
    spot = float(technicals.get("spot", 0))
    earnings_buffer = int(rec_config.get("earnings_buffer_days", 7))
    delta_min = float(rec_config.get("delta_min", 0.10))
    delta_max = float(rec_config.get("delta_max", 0.25))
    resistance_buffer = float(rec_config.get("resistance_pct_buffer", 0.02))

    empty_row = _make_base_row(ticker, term_label, spot, min_acceptable_price)

    if not call_candidates:
        empty_row["reason"] = f"No {term_label} CALL candidates available"
        return [empty_row]

    def _delta_ok(c: Dict) -> bool:
        d = c.get("delta")
        return d is not None and delta_min <= abs(float(d)) <= delta_max

    def _earnings_ok(c: Dict) -> bool:
        if earnings_date is None:
            return True
        c_exp_str = c.get("expiration")
        if not c_exp_str:
            return True
        try:
            c_exp = date.fromisoformat(str(c_exp_str))
        except ValueError:
            return True
        days_before = (c_exp - earnings_date).days
        return not (earnings_date <= c_exp and days_before <= earnings_buffer)

    # Prefer delta-qualified candidates; fill remaining slots from the rest.
    by_score = lambda lst: sorted(lst, key=lambda x: float(x.get("score") or 0), reverse=True)
    delta_ok = by_score([c for c in call_candidates if _delta_ok(c)])
    delta_out = by_score([c for c in call_candidates if not _delta_ok(c)])
    top_n = (delta_ok + delta_out)[:max_suggestions]

    results: List[Dict[str, Any]] = []
    for best in top_n:
        strike = float(best["strike"])
        premium = float(best.get("mid", 0))
        delta_val = best.get("delta")
        dte_val = best.get("dte")

        near_round = _near_round_number(strike)
        near_res = _near_resistance(strike, resistance, resistance_buffer)
        below_min = min_acceptable_price is not None and strike < min_acceptable_price
        delta_in_range = _delta_ok(best)
        earnings_ok = _earnings_ok(best)

        # IVR — informational only, does not affect verdict
        best_iv = float(best["implied_volatility"]) if best.get("implied_volatility") else None
        ivr_value, ivr_source = compute_ivr_proxy(price_df, best_iv)

        # ── Verdict ────────────────────────────────────────────────────────
        issues: List[str] = []
        if not delta_in_range:
            d = abs(float(delta_val)) if delta_val is not None else None
            if d is not None:
                issues.append(f"delta {d:.2f} outside target {delta_min:.2f}–{delta_max:.2f}")
            else:
                issues.append("delta unavailable")
        if not earnings_ok:
            issues.append("earnings too close to expiration")
        if below_min:
            issues.append(f"strike ${strike:.2f} below min ${min_acceptable_price:.2f}")

        if issues:
            verdict = "Borderline"
            reason = "; ".join(issues)
        else:
            verdict = "Yes"
            d_str = f"{abs(float(delta_val)):.2f}" if delta_val is not None else "n/a"
            reason = f"delta {d_str}"
            if near_res:
                reason += "; strike near resistance (favourable)"

        max_profit = round((strike - spot + premium) * 100, 2) if spot > 0 else None
        downside_breakeven = round(spot - premium, 2) if spot > 0 else None

        row = _make_base_row(ticker, term_label, spot, min_acceptable_price)
        row.update({
            "recommend": verdict,
            "reason": reason,
            "strike": strike,
            "expiration": best.get("expiration"),
            "premium": round(premium, 2),
            "delta": round(float(delta_val), 3) if delta_val is not None else None,
            "ivr": ivr_value,
            "ivr_source": ivr_source,
            "max_profit": max_profit,
            "downside_breakeven": downside_breakeven,
            "near_resistance": near_res,
            "near_round_number": near_round,
            "below_min_price": below_min,
            "dte": dte_val,
            "annualized_yield": best.get("annualized_yield"),
        })
        results.append(row)

    return results


# ── Per-ticker entry point ──────────────────────────────────────────────────

def _resolve_cc_buckets(rec_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the ordered list of CC term buckets from config.

    Near-term buckets (config-driven) come first; Medium/Long-Term are appended
    when keep_medium_long is true. Each bucket is {label, min_dte, max_dte} inclusive.
    """
    near_term = rec_config.get("near_term_buckets") or DEFAULT_CC_REC_CONFIG["near_term_buckets"]
    buckets = [dict(b) for b in near_term]
    if rec_config.get("keep_medium_long", True):
        buckets.extend(dict(b) for b in _MEDIUM_LONG_BUCKETS)
    return buckets


def recommend_cc_for_ticker(
    ticker: str,
    candidates: List[Dict[str, Any]],
    price_df: pd.DataFrame,
    technicals: Dict[str, float],
    earnings_date: Optional[date],
    min_acceptable_price: Optional[float],
    rec_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Produce CC suggestions for one ticker across the configured term buckets
    (near-term ~1d/~3d/~5d, plus Medium/Long-Term when keep_medium_long is on).
    Returns a flat list of suggestion dicts, ≥1 per term.
    """
    buckets = _resolve_cc_buckets(rec_config)
    spot = float(technicals.get("spot", 0))
    if spot <= 0:
        no_data = _make_base_row(ticker, "", spot, min_acceptable_price)
        no_data["reason"] = "Spot price unavailable"
        return [{**no_data, "term": b["label"]} for b in buckets]

    max_suggestions = int(rec_config.get("max_suggestions_per_term", 3))

    all_calls = [c for c in candidates if c.get("strategy") == "CALL"]

    resistance = get_resistance_levels(price_df)

    results: List[Dict[str, Any]] = []
    for bucket in buckets:
        lo = int(bucket.get("min_dte", 0))
        hi = int(bucket.get("max_dte", 10_000))
        pool = [c for c in all_calls if lo <= (c.get("dte") if c.get("dte") is not None else -1) <= hi]
        results.extend(
            _recommend_for_bucket(
                ticker=ticker,
                term_label=bucket["label"],
                call_candidates=pool,
                resistance=resistance,
                technicals=technicals,
                earnings_date=earnings_date,
                min_acceptable_price=min_acceptable_price,
                price_df=price_df,
                rec_config=rec_config,
                max_suggestions=max_suggestions,
            )
        )
    return results


# ── Batch entry point ───────────────────────────────────────────────────────

def build_cc_recommendations(
    ticker_results: Dict[str, Dict[str, Any]],
    cc_tickers: List[str],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Run CC recommendation engine for all covered-call tickers.

    When best_only is on, only "Yes" verdicts are kept and at most max_per_term
    rows are retained per (ticker, term) — the "best recommendations only" view.
    Otherwise all suggestions are returned (legacy verbose behaviour).
    Results are ordered near-term → Medium → Long, then by yield desc.
    Capped at max_recommendations.
    """
    rec_config = config.get("cc_recommendation", {})
    if not rec_config.get("enabled", True):
        return []

    max_recs = int(rec_config.get("max_recommendations", 50))
    best_only = bool(rec_config.get("best_only", False))
    max_per_term = int(rec_config.get("max_per_term", 1))
    min_prices: Dict[str, Any] = rec_config.get("min_acceptable_sale_prices") or {}

    # Term ordering follows the configured bucket order (near-term first).
    buckets = _resolve_cc_buckets(rec_config)
    term_order = {b["label"]: i for i, b in enumerate(buckets)}

    results: List[Dict[str, Any]] = []
    for ticker in cc_tickers:
        tr = ticker_results.get(ticker, {})
        raw_min = min_prices.get(ticker) or min_prices.get(ticker.upper())
        min_price = float(raw_min) if raw_min is not None else None
        recs = recommend_cc_for_ticker(
            ticker=ticker,
            candidates=tr.get("candidates", []),
            price_df=tr.get("price_df", pd.DataFrame()),
            technicals=tr.get("technicals", {}),
            earnings_date=tr.get("earnings_date"),
            min_acceptable_price=min_price,
            rec_config=rec_config,
        )

        if best_only:
            # Keep only "Yes" verdicts, at most max_per_term per (ticker, term),
            # ranked by annualized yield desc. Tickers/terms with no "Yes" are dropped.
            by_term: Dict[str, List[Dict[str, Any]]] = {}
            for r in recs:
                if r.get("recommend") == "Yes":
                    by_term.setdefault(r.get("term", ""), []).append(r)
            for term, rows in by_term.items():
                rows.sort(key=lambda x: float(x.get("annualized_yield") or 0), reverse=True)
                results.extend(rows[:max_per_term])
        else:
            results.extend(recs)

    results.sort(
        key=lambda r: (
            r["ticker"],
            term_order.get(r.get("term", ""), 99),
            -(float(r.get("annualized_yield") or 0)),
        )
    )
    return results[:max_recs]
