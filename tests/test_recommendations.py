"""Contract tests for the simplified recommendation report.

Covers the behaviour the "best recommendations + near-term covered calls" change
depends on:
  - near-term CC DTE bucketing (~1d / ~3d / ~5d) + Medium/Long-Term
  - CC best_only filter (Yes verdicts only, top-1 per ticker per term)
  - CSP single-best-per-ticker collapse
  - report output shape (no detailed screening sections; recommendation CSV schema)

All tests are deterministic and network-free.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from agent.recommendation.cc_recommender import build_cc_recommendations
from agent.recommendation.csp_recommender import build_csp_recommendations
from agent.reporting.render import write_reports


def _cc_config(**overrides):
    cfg = {
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
        "min_acceptable_sale_prices": {},
    }
    cfg.update(overrides)
    return {"cc_recommendation": cfg}


def _csp_config(**overrides):
    cfg = {
        "enabled": True,
        "max_recommendations": 30,
        "best_per_ticker": True,
        "ivr_min": 0.0,
        "earnings_buffer_days": 7,
        "delta_min": 0.10,
        "delta_max": 0.25,
        "use_support_filter": False,
        "support_pct_buffer": 0.02,
    }
    cfg.update(overrides)
    return {"csp_recommendation": cfg}


def _call_cand(dte, strike, ayield, score, delta=0.20):
    exp = (date.today() + timedelta(days=dte)).isoformat()
    return {
        "strategy": "CALL",
        "dte": dte,
        "strike": strike,
        "mid": 2.0,
        "delta": delta,
        "expiration": exp,
        "implied_volatility": 0.30,
        "score": score,
        "annualized_yield": ayield,
    }


def _put_cand(dte, strike, ayield, score, delta=-0.20):
    exp = (date.today() + timedelta(days=dte)).isoformat()
    return {
        "strategy": "PUT",
        "dte": dte,
        "strike": strike,
        "mid": 2.0,
        "delta": delta,
        "expiration": exp,
        "implied_volatility": 0.30,
        "score": score,
        "annualized_yield": ayield,
    }


def _price_df(n=60, price=100.0):
    idx = pd.date_range(end=date.today(), periods=n, freq="D")
    close = pd.Series([price + (i % 5) for i in range(n)], index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close}
    )


def _ticker_results(ticker, candidates):
    return {
        ticker: {
            "candidates": candidates,
            "price_df": _price_df(),
            "technicals": {"spot": 100.0},
            "earnings_date": None,
        }
    }


# ── CC bucketing ────────────────────────────────────────────────────────────

def test_cc_near_term_and_medium_long_bucketing():
    cands = [
        _call_cand(1, 101, 0.30, 0.9),   # ~1 Day
        _call_cand(3, 102, 0.31, 0.9),   # ~3 Day
        _call_cand(5, 103, 0.32, 0.9),   # ~5 Day
        _call_cand(20, 105, 0.33, 0.9),  # Medium-Term
        _call_cand(40, 110, 0.34, 0.9),  # Long-Term
    ]
    recs = build_cc_recommendations(_ticker_results("TEST", cands), ["TEST"], _cc_config())

    terms = [r["term"] for r in recs]
    assert terms == ["~1 Day", "~3 Day", "~5 Day", "Medium-Term", "Long-Term"]
    assert all(r["recommend"] == "Yes" for r in recs)


def test_cc_gap_between_near_term_and_medium_is_dropped():
    # 7-14 DTE is an intentional gap: no near-term bucket and below Medium-Term (15+).
    cands = [_call_cand(10, 104, 0.30, 0.9)]
    recs = build_cc_recommendations(_ticker_results("TEST", cands), ["TEST"], _cc_config())
    assert recs == []


def test_cc_keep_medium_long_disabled_drops_longer_terms():
    cands = [_call_cand(1, 101, 0.30, 0.9), _call_cand(40, 110, 0.34, 0.9)]
    recs = build_cc_recommendations(
        _ticker_results("TEST", cands), ["TEST"], _cc_config(keep_medium_long=False)
    )
    assert [r["term"] for r in recs] == ["~1 Day"]


# ── CC best_only filter ─────────────────────────────────────────────────────

def test_cc_best_only_keeps_top1_yes_per_term():
    cands = [
        _call_cand(3, 102, 0.20, 0.8, delta=0.20),  # Yes, lower yield
        _call_cand(3, 103, 0.40, 0.7, delta=0.20),  # Yes, higher yield  <- expected winner
        _call_cand(3, 104, 0.99, 0.9, delta=0.50),  # Borderline (delta out of range), dropped
    ]
    recs = build_cc_recommendations(_ticker_results("TEST", cands), ["TEST"], _cc_config())

    assert len(recs) == 1
    assert recs[0]["term"] == "~3 Day"
    assert recs[0]["recommend"] == "Yes"
    assert recs[0]["annualized_yield"] == 0.40


def test_cc_best_only_drops_ticker_with_no_yes():
    cands = [_call_cand(3, 104, 0.99, 0.9, delta=0.50)]  # only Borderline
    recs = build_cc_recommendations(_ticker_results("TEST", cands), ["TEST"], _cc_config())
    assert recs == []


# ── CSP single best per ticker ──────────────────────────────────────────────

def test_csp_best_per_ticker_returns_single_row():
    cands = [
        _put_cand(10, 95, 0.20, 0.8),
        _put_cand(20, 90, 0.35, 0.9),  # highest yield
        _put_cand(40, 85, 0.25, 0.7),
    ]
    recs = build_csp_recommendations(_ticker_results("TEST", cands), ["TEST"], _csp_config())

    assert len(recs) == 1
    assert recs[0]["ticker"] == "TEST"
    assert recs[0]["annualized_yield"] == 0.35


def test_csp_best_per_ticker_one_row_each_ticker():
    tr = {}
    tr.update(_ticker_results("AAA", [_put_cand(10, 95, 0.20, 0.8)]))
    tr.update(_ticker_results("BBB", [_put_cand(20, 90, 0.30, 0.9)]))
    recs = build_csp_recommendations(tr, ["AAA", "BBB"], _csp_config())

    assert sorted(r["ticker"] for r in recs) == ["AAA", "BBB"]
    assert len(recs) == 2


# ── Report output shape ─────────────────────────────────────────────────────

_EXPECTED_CSV_HEADER = (
    "type,ticker,term,recommend,expiration,dte,spot,strike,premium,"
    "annualized_yield,delta,ivr,max_profit,breakeven,reason"
)


def test_write_reports_has_rec_tables_and_no_detailed_sections(tmp_path):
    cc = build_cc_recommendations(
        _ticker_results("TEST", [_call_cand(3, 102, 0.30, 0.9)]), ["TEST"], _cc_config()
    )
    csp = build_csp_recommendations(
        _ticker_results("TEST", [_put_cand(20, 90, 0.30, 0.9)]), ["TEST"], _csp_config()
    )

    config = {"output_dir": str(tmp_path)}
    csv_path, html_path = write_reports(
        [], config, "disclaimer",
        csp_recommendations=csp, cc_recommendations=cc,
    )

    html = open(html_path, encoding="utf-8").read()
    assert "Covered Call Recommendations" in html
    assert "CSP Recommendations" in html
    # Detailed per-ticker screening sections must be gone.
    assert "section-puts" not in html
    assert "section-calls" not in html
    assert "ticker-block" not in html

    csv_header = open(csv_path, encoding="utf-8").read().splitlines()[0].strip()
    assert csv_header == _EXPECTED_CSV_HEADER


def test_write_reports_empty_cc_shows_note(tmp_path):
    config = {"output_dir": str(tmp_path)}
    _, html_path = write_reports(
        [], config, "disclaimer", csp_recommendations=[], cc_recommendations=[],
    )
    html = open(html_path, encoding="utf-8").read()
    assert "No qualifying near-term covered calls today." in html
