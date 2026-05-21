"""
engine.py - Pine Bank Equity Research Engine
Specialized analysis for Banco Pine (PINE4) with banking-specific metrics
"""
import warnings
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
np.random.seed(42)


# ============================================================
# PINE BANK UNIVERSE — Banco Pine + Peer Group
# ============================================================

PINE_TICKER = "PINE4.SA"

# Peers: middle-market and corporate banks for fair comparison
PEER_GROUP = {
    "PINE4.SA":  ("Banco Pine",      "Middle Market"),
    "ABCB4.SA":  ("Banco ABC Brasil","Middle Market"),
    "BPAC11.SA": ("BTG Pactual",     "Investment Bank"),
    "BMGB4.SA":  ("Banco BMG",       "Consumer Credit"),
    "BRSR6.SA":  ("Banrisul",        "Regional"),
    "ITUB4.SA":  ("Itau Unibanco",   "Universal Bank"),
    "BBDC4.SA":  ("Bradesco",        "Universal Bank"),
    "BBAS3.SA":  ("Banco do Brasil", "Universal Bank"),
    "SANB11.SA": ("Santander BR",    "Universal Bank"),
}


# ============================================================
# UTILITIES
# ============================================================

def safe_div(a, b):
    try:
        a, b = float(a), float(b)
        if np.isnan(a) or np.isnan(b) or b == 0:
            return np.nan
        return a / b
    except Exception:
        return np.nan


def to_float(v):
    if v is None:
        return np.nan
    try:
        f = float(v)
        return f if not (np.isnan(f) or np.isinf(f)) else np.nan
    except (TypeError, ValueError):
        return np.nan


def to_str(v, default="-"):
    if v is None:
        return default
    s = str(v).strip()
    if not s or s.lower() in ("none", "nan", "?", "n/a", "null"):
        return default
    return s


def get_financial_value(df, possible_keys):
    """Try multiple possible row names in a financial statement DataFrame."""
    if df is None or df.empty:
        return np.nan
    for key in possible_keys:
        if key in df.index:
            val = to_float(df.loc[key].iloc[0])
            if pd.notna(val):
                return val
    return np.nan


# ============================================================
# TOOL 1: PRICES & MARKET METRICS
# ============================================================

def fetch_prices(tickers, start="2020-01-01"):
    """Download adjusted close prices and compute market metrics."""
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])
    p = raw.stack(future_stack=True).reset_index()
    p.columns = ["data", "ticker", "preco"]
    p["data"] = pd.to_datetime(p["data"])
    p = p.dropna(subset=["preco"]).sort_values(["ticker", "data"])
    g = p.groupby("ticker")
    p["ret_dia"] = g["preco"].pct_change()
    p["vol_21"] = g["ret_dia"].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    p["mom_3m"] = g["preco"].transform(lambda x: x.pct_change(63))
    p["mom_6m"] = g["preco"].transform(lambda x: x.pct_change(126))
    p["mom_12m"] = g["preco"].transform(lambda x: x.pct_change(252))
    p["drawdown"] = g["preco"].transform(lambda x: (x - x.cummax()) / x.cummax())
    p["sma_20"] = g["preco"].transform(lambda x: x.rolling(20).mean())
    p["sma_60"] = g["preco"].transform(lambda x: x.rolling(60).mean())
    p["sma_200"] = g["preco"].transform(lambda x: x.rolling(200).mean())
    return p


# ============================================================
# TOOL 2: FUNDAMENTALS & BANKING METRICS
# ============================================================

def fetch_bank_fundamentals(tickers, prices_df=None):
    """
    Fetch comprehensive bank fundamentals with multiple fallback layers.
    Includes banking-specific metrics: ROE, ROA, NIM proxy, P/B, etc.
    """
    rows = []
    for idx, t in enumerate(tickers):
        nome_default, segment_default = PEER_GROUP.get(t, (t.replace(".SA", ""), "-"))
        r = {
            "ticker": t,
            "nome": nome_default,
            "segment": segment_default,
            "preco": np.nan, "marketCap": np.nan, "shares": np.nan,
            "P_L": np.nan, "P_VP": np.nan,
            "EV": np.nan, "totalAssets": np.nan, "totalEquity": np.nan,
            "netIncome": np.nan, "totalRevenue": np.nan,
            "totalDebt": np.nan, "div_yield": np.nan,
            "profitMargins": np.nan, "returnOnEquity": np.nan,
            "returnOnAssets": np.nan, "revenueGrowth": np.nan,
            "debtToEquity": np.nan, "operatingMargins": np.nan,
            "leverage_ratio": np.nan,  # Assets / Equity (bank leverage)
            "asset_turnover": np.nan,   # Revenue / Assets (NIM proxy)
            "earnings_yield": np.nan,   # 1 / P_L
            "summary": "",
        }

        # ===== Layer 1: .info =====
        info = {}
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            pass

        if info:
            r["nome"] = to_str(info.get("shortName") or info.get("longName"), r["nome"])
            r["preco"] = to_float(info.get("currentPrice")
                                   or info.get("regularMarketPrice")
                                   or info.get("previousClose"))
            r["marketCap"] = to_float(info.get("marketCap"))
            r["shares"] = to_float(info.get("sharesOutstanding") or info.get("floatShares"))
            r["P_L"] = to_float(info.get("trailingPE") or info.get("forwardPE"))
            r["P_VP"] = to_float(info.get("priceToBook"))
            r["EV"] = to_float(info.get("enterpriseValue"))
            r["totalDebt"] = to_float(info.get("totalDebt"))
            r["div_yield"] = to_float(info.get("dividendYield"))
            r["profitMargins"] = to_float(info.get("profitMargins"))
            r["returnOnEquity"] = to_float(info.get("returnOnEquity"))
            r["returnOnAssets"] = to_float(info.get("returnOnAssets"))
            r["revenueGrowth"] = to_float(info.get("revenueGrowth"))
            r["operatingMargins"] = to_float(info.get("operatingMargins"))
            r["debtToEquity"] = to_float(info.get("debtToEquity"))
            r["summary"] = to_str(info.get("longBusinessSummary"), "")

        # ===== Layer 2: fast_info =====
        if pd.isna(r["preco"]) or pd.isna(r["marketCap"]):
            try:
                tk = yf.Ticker(t)
                fi = tk.fast_info
                if pd.isna(r["preco"]):
                    r["preco"] = to_float(getattr(fi, "last_price", None)
                                           or getattr(fi, "previous_close", None))
                if pd.isna(r["marketCap"]):
                    r["marketCap"] = to_float(getattr(fi, "market_cap", None))
                if pd.isna(r["shares"]):
                    r["shares"] = to_float(getattr(fi, "shares", None))
            except Exception:
                pass

        # ===== Layer 3: prices_df fallback =====
        if pd.isna(r["preco"]) and prices_df is not None:
            try:
                sub = prices_df[prices_df["ticker"] == t]
                if not sub.empty:
                    r["preco"] = to_float(sub["preco"].iloc[-1])
            except Exception:
                pass

        # ===== Layer 4: Financial statements =====
        try:
            tk = yf.Ticker(t)
            fin = tk.financials
            bs = tk.balance_sheet

            # Net Income
            r["netIncome"] = get_financial_value(fin, [
                "Net Income", "Net Income Common Stockholders",
                "Net Income Continuous Operations", "Net Income From Continuing Operations",
            ])
            # Total Revenue
            r["totalRevenue"] = get_financial_value(fin, [
                "Total Revenue", "Operating Revenue", "Total Revenues",
            ])
            # Total Equity (book value)
            r["totalEquity"] = get_financial_value(bs, [
                "Stockholders Equity", "Total Equity Gross Minority Interest",
                "Common Stock Equity", "Total Stockholders Equity",
            ])
            # Total Assets (key for banks)
            r["totalAssets"] = get_financial_value(bs, [
                "Total Assets", "Assets",
            ])

            # Compute missing multiples
            if pd.isna(r["P_L"]) and pd.notna(r["marketCap"]) and pd.notna(r["netIncome"]) and r["netIncome"] > 0:
                r["P_L"] = safe_div(r["marketCap"], r["netIncome"])
            if pd.isna(r["P_VP"]) and pd.notna(r["marketCap"]) and pd.notna(r["totalEquity"]) and r["totalEquity"] > 0:
                r["P_VP"] = safe_div(r["marketCap"], r["totalEquity"])
            if pd.isna(r["returnOnEquity"]) and pd.notna(r["netIncome"]) and pd.notna(r["totalEquity"]) and r["totalEquity"] > 0:
                r["returnOnEquity"] = safe_div(r["netIncome"], r["totalEquity"])
            if pd.isna(r["returnOnAssets"]) and pd.notna(r["netIncome"]) and pd.notna(r["totalAssets"]) and r["totalAssets"] > 0:
                r["returnOnAssets"] = safe_div(r["netIncome"], r["totalAssets"])
            if pd.isna(r["profitMargins"]) and pd.notna(r["netIncome"]) and pd.notna(r["totalRevenue"]) and r["totalRevenue"] > 0:
                r["profitMargins"] = safe_div(r["netIncome"], r["totalRevenue"])
        except Exception:
            pass

        # Derived metrics
        if pd.notna(r["totalAssets"]) and pd.notna(r["totalEquity"]) and r["totalEquity"] > 0:
            r["leverage_ratio"] = safe_div(r["totalAssets"], r["totalEquity"])
        if pd.notna(r["totalRevenue"]) and pd.notna(r["totalAssets"]) and r["totalAssets"] > 0:
            r["asset_turnover"] = safe_div(r["totalRevenue"], r["totalAssets"])
        if pd.notna(r["P_L"]) and r["P_L"] > 0:
            r["earnings_yield"] = 1.0 / r["P_L"]

        rows.append(r)

        if idx < len(tickers) - 1:
            time.sleep(0.25)

    return pd.DataFrame(rows)


# ============================================================
# TOOL 3: DETAILED FINANCIAL STATEMENTS FOR PINE
# ============================================================

def fetch_financial_statements(ticker):
    """
    Fetch full financial statements (income, balance, cashflow) for detailed analysis.
    Returns dict with 'income', 'balance', 'cashflow' DataFrames.
    """
    result = {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}
    try:
        tk = yf.Ticker(ticker)
        # Annual statements
        result["income"] = tk.financials if tk.financials is not None else pd.DataFrame()
        result["balance"] = tk.balance_sheet if tk.balance_sheet is not None else pd.DataFrame()
        result["cashflow"] = tk.cashflow if tk.cashflow is not None else pd.DataFrame()
        # Quarterly (additional detail)
        result["income_q"] = tk.quarterly_financials if tk.quarterly_financials is not None else pd.DataFrame()
        result["balance_q"] = tk.quarterly_balance_sheet if tk.quarterly_balance_sheet is not None else pd.DataFrame()
    except Exception:
        pass
    return result


def compute_bank_kpis(statements):
    """
    Compute banking-specific KPIs from financial statements.
    Returns DataFrame with KPIs per year.
    """
    inc = statements.get("income", pd.DataFrame())
    bs = statements.get("balance", pd.DataFrame())
    if inc.empty or bs.empty:
        return pd.DataFrame()

    # Get common columns (years)
    common_cols = sorted(set(inc.columns) & set(bs.columns), reverse=True)
    if not common_cols:
        return pd.DataFrame()

    kpis = []
    for col in common_cols:
        year = col.year if hasattr(col, "year") else str(col)[:4]
        row = {"year": year}

        # Income statement
        ni = get_financial_value(inc[[col]], [
            "Net Income", "Net Income Common Stockholders",
            "Net Income Continuous Operations",
        ])
        rev = get_financial_value(inc[[col]], ["Total Revenue", "Operating Revenue"])
        op_inc = get_financial_value(inc[[col]], ["Operating Income", "Operating Revenue"])
        interest_inc = get_financial_value(inc[[col]], [
            "Interest Income", "Interest Income Non Operating",
        ])

        # Balance sheet
        equity = get_financial_value(bs[[col]], [
            "Stockholders Equity", "Total Equity Gross Minority Interest",
            "Common Stock Equity",
        ])
        assets = get_financial_value(bs[[col]], ["Total Assets", "Assets"])
        debt = get_financial_value(bs[[col]], [
            "Total Debt", "Long Term Debt", "Net Debt",
        ])

        row["net_income"] = ni
        row["revenue"] = rev
        row["operating_income"] = op_inc
        row["total_assets"] = assets
        row["total_equity"] = equity
        row["total_debt"] = debt

        # KPIs
        row["ROE"] = safe_div(ni, equity)
        row["ROA"] = safe_div(ni, assets)
        row["net_margin"] = safe_div(ni, rev)
        row["leverage"] = safe_div(assets, equity)
        row["debt_to_equity"] = safe_div(debt, equity)
        # NIM proxy = Revenue / Average Assets (proxy for net interest margin)
        row["NIM_proxy"] = safe_div(rev, assets)

        kpis.append(row)

    return pd.DataFrame(kpis)


# ============================================================
# TOOL 4: ML — Random Forest Walk-Forward
# ============================================================

def run_ml(prices, fund_df=None):
    feat = prices.copy()
    feat["month"] = feat["data"].dt.to_period("M")
    me = feat.sort_values("data").groupby(["ticker", "month"]).tail(1).copy()
    me = me.sort_values(["ticker", "data"])

    mkt = ["vol_21", "mom_3m", "mom_6m", "mom_12m", "drawdown"]
    for c in mkt:
        me[f"{c}_z"] = me.groupby("data")[c].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1))
    mkt_z = [f"{c}_z" for c in mkt]

    fund_cols = []
    if fund_df is not None and not fund_df.empty:
        fund_feats = ["P_L", "P_VP", "returnOnEquity", "returnOnAssets",
                      "profitMargins", "revenueGrowth", "leverage_ratio"]
        fund_avail = [c for c in fund_feats if c in fund_df.columns]
        if fund_avail:
            me = me.merge(fund_df[["ticker"] + fund_avail], on="ticker", how="left")
            for c in fund_avail:
                me[f"{c}_z"] = me.groupby("data")[c].transform(
                    lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1))
                me[f"{c}_z"] = me[f"{c}_z"].fillna(0)
            fund_cols = [f"{c}_z" for c in fund_avail]

    me["ret_12m_fwd"] = me.groupby("ticker")["preco"].transform(
        lambda s: s.shift(-12) / s - 1.0)

    core = mkt + mkt_z + fund_cols
    data_ml = me.dropna(subset=mkt + mkt_z + ["ret_12m_fwd"]).copy()
    for c in fund_cols:
        if c not in data_ml.columns:
            data_ml[c] = 0
    data_ml["year"] = data_ml["data"].dt.year

    yrs = sorted(data_ml["year"].unique())
    metrics, fis = [], []
    for ty in yrs[2:]:
        tr = data_ml[data_ml["year"] < ty]
        te = data_ml[data_ml["year"] == ty]
        if len(tr) < 20 or len(te) < 3:
            continue
        m = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                   random_state=42, n_jobs=-1)
        m.fit(tr[core].values, tr["ret_12m_fwd"].values)
        yp = m.predict(te[core].values)
        yt = te["ret_12m_fwd"].values
        sp = spearmanr(yt, yp).correlation if len(set(yt)) > 1 else np.nan
        metrics.append({
            "year": ty,
            "rmse": np.sqrt(mean_squared_error(yt, yp)),
            "mae": mean_absolute_error(yt, yp),
            "r2": r2_score(yt, yp),
            "spearman_ic": sp, "n": len(te),
        })
        fis.append(pd.Series(m.feature_importances_, index=core, name=f"y{ty}"))

    metrics_df = pd.DataFrame(metrics)
    fi_df = pd.concat(fis, axis=1).T if fis else pd.DataFrame()

    latest = me.dropna(subset=mkt + mkt_z).sort_values("data").groupby("ticker").tail(1).copy()
    for c in fund_cols:
        if c not in latest.columns:
            latest[c] = 0
        else:
            latest[c] = latest[c].fillna(0)
    if len(data_ml) >= 20:
        train_all = data_ml.dropna(subset=mkt + mkt_z + ["ret_12m_fwd"])
        mfin = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=3,
                                      random_state=42, n_jobs=-1)
        mfin.fit(train_all[core].values, train_all["ret_12m_fwd"].values)
        latest["pred_ret_12m"] = mfin.predict(latest[core].values)
        latest["rank_pred"] = latest["pred_ret_12m"].rank(ascending=False, method="min").astype(int)
    else:
        latest["pred_ret_12m"] = np.nan
        latest["rank_pred"] = np.nan

    return metrics_df, fi_df, latest


# ============================================================
# TOOL 5: TRADING SYSTEM (MA Crossover)
# ============================================================

def run_trading_system(prices, ma_short=20, ma_long=60):
    all_res, summary = [], []
    for ticker in prices["ticker"].unique():
        sub = prices[prices["ticker"] == ticker][["data", "preco"]].copy().sort_values("data")
        if len(sub) < ma_long + 10:
            continue
        df = pd.DataFrame({"data": sub["data"].values, "preco": sub["preco"].values})
        df["ma_short"] = df["preco"].rolling(ma_short).mean()
        df["ma_long"] = df["preco"].rolling(ma_long).mean()
        df["signal"] = np.where(df["ma_short"] > df["ma_long"], 1, 0)
        df["ret_acao"] = df["preco"].pct_change()
        df["ret_est"] = df["signal"].shift(1) * df["ret_acao"]
        df = df.dropna()
        df["acum_acao"] = (1 + df["ret_acao"]).cumprod()
        df["acum_est"] = (1 + df["ret_est"]).cumprod()
        df["ticker"] = ticker
        all_res.append(df)
        rb = df["acum_acao"].iloc[-1] - 1
        rs = df["acum_est"].iloc[-1] - 1
        summary.append({
            "ticker": ticker,
            "ret_buyhold": round(rb * 100, 1),
            "ret_estrategia": round(rs * 100, 1),
            "alpha": round((rs - rb) * 100, 1),
        })
    curves = pd.concat(all_res, ignore_index=True) if all_res else pd.DataFrame()
    return curves, pd.DataFrame(summary)


# ============================================================
# TOOL 6: PEER COMPARISON & RANKING
# ============================================================

def run_peer_comparison(fund_df):
    """Rank Pine vs peers across key bank metrics."""
    df = fund_df.copy()
    # Score each metric (higher = better, except P_L, P_VP, leverage)
    metrics_higher_better = ["returnOnEquity", "returnOnAssets", "profitMargins",
                              "revenueGrowth", "earnings_yield"]
    metrics_lower_better = ["P_L", "P_VP", "leverage_ratio"]

    for m in metrics_higher_better:
        if m in df.columns:
            df[f"{m}_rank"] = df[m].rank(ascending=False, method="min")
            df[f"{m}_pct"] = df[m].rank(pct=True) * 100
    for m in metrics_lower_better:
        if m in df.columns:
            df[f"{m}_rank"] = df[m].rank(ascending=True, method="min")
            df[f"{m}_pct"] = df[m].rank(pct=True, ascending=False) * 100

    # Composite score
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    if pct_cols:
        df["composite_pct"] = df[pct_cols].mean(axis=1)
        df["overall_rank"] = df["composite_pct"].rank(ascending=False, method="min").astype(int)
    return df


# ============================================================
# TOOL 7: VALUATION SCENARIOS
# ============================================================

def run_valuation_scenarios(pine_fund, peers_fund):
    """
    Compute fair value scenarios for Pine using peer multiples.
    Scenarios: bear (peer p25), base (peer median), bull (peer p75).
    """
    # Get Pine row
    pine = peers_fund[peers_fund["ticker"] == PINE_TICKER].iloc[0] if (
        PINE_TICKER in peers_fund["ticker"].values
    ) else None
    if pine is None:
        return pd.DataFrame()

    peers_only = peers_fund[peers_fund["ticker"] != PINE_TICKER].copy()

    scenarios = []
    for mult_name in ["P_L", "P_VP"]:
        if mult_name not in peers_only.columns:
            continue
        peer_mult = peers_only[mult_name].dropna()
        if peer_mult.empty:
            continue
        if mult_name == "P_L":
            base_metric = pine.get("netIncome") or (
                pine.get("marketCap") / pine.get("P_L") if pd.notna(pine.get("P_L")) else np.nan
            )
        else:  # P_VP
            base_metric = pine.get("totalEquity") or (
                pine.get("marketCap") / pine.get("P_VP") if pd.notna(pine.get("P_VP")) else np.nan
            )
        if pd.isna(base_metric) or base_metric <= 0:
            continue

        bear_mult = peer_mult.quantile(0.25)
        base_mult = peer_mult.median()
        bull_mult = peer_mult.quantile(0.75)

        shares = pine.get("shares", np.nan)
        if pd.isna(shares):
            continue
        current_price = pine.get("preco", np.nan)

        for label, mult in [("Bear (p25)", bear_mult),
                             ("Base (median)", base_mult),
                             ("Bull (p75)", bull_mult)]:
            implied_mcap = base_metric * mult
            implied_price = implied_mcap / shares
            upside = (implied_price / current_price - 1) * 100 if pd.notna(current_price) and current_price > 0 else np.nan
            scenarios.append({
                "multiple": mult_name,
                "scenario": label,
                "peer_multiple": round(mult, 2),
                "implied_price": round(implied_price, 2),
                "current_price": round(current_price, 2) if pd.notna(current_price) else np.nan,
                "upside_pct": round(upside, 1) if pd.notna(upside) else np.nan,
            })

    return pd.DataFrame(scenarios)


# ============================================================
# TOOL 8: MONTE CARLO PRICE SIMULATION
# ============================================================

def run_price_simulation(prices, ticker=PINE_TICKER, days=252, n_sim=1000):
    """
    Monte Carlo simulation of Pine stock price over the next year.
    Uses GBM with parameters estimated from historical data.
    """
    sub = prices[prices["ticker"] == ticker].dropna(subset=["preco", "ret_dia"]).copy()
    if len(sub) < 100:
        return pd.DataFrame(), {}

    mu = sub["ret_dia"].mean()
    sigma = sub["ret_dia"].std()
    s0 = sub["preco"].iloc[-1]

    # GBM simulation
    np.random.seed(42)
    sims = np.zeros((n_sim, days))
    for i in range(n_sim):
        prices_sim = [s0]
        for _ in range(days - 1):
            z = np.random.normal()
            prices_sim.append(prices_sim[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * z))
        sims[i] = prices_sim

    # Stats
    final_prices = sims[:, -1]
    stats = {
        "current_price": round(s0, 2),
        "mean_1y": round(final_prices.mean(), 2),
        "median_1y": round(np.median(final_prices), 2),
        "p5_1y": round(np.percentile(final_prices, 5), 2),
        "p95_1y": round(np.percentile(final_prices, 95), 2),
        "prob_gain": round((final_prices > s0).mean() * 100, 1),
        "prob_20pct_gain": round((final_prices > s0 * 1.2).mean() * 100, 1),
        "prob_20pct_loss": round((final_prices < s0 * 0.8).mean() * 100, 1),
        "expected_return": round((final_prices.mean() / s0 - 1) * 100, 1),
    }

    # Build paths DataFrame (sample of 100 paths for visualization)
    last_date = sub["data"].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=days, freq="B")
    paths_df = pd.DataFrame(sims[:100].T, index=future_dates)
    paths_df.columns = [f"sim_{i}" for i in range(100)]

    return paths_df, stats
