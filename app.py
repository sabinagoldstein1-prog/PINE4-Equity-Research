"""
app.py — Pine Bank Equity Research Terminal
Single-asset deep dive analysis: PINE4 vs peer group
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from engine import (
    PINE_TICKER, PEER_GROUP,
    fetch_prices, fetch_bank_fundamentals,
    fetch_financial_statements, compute_bank_kpis,
    run_ml, run_trading_system, run_peer_comparison,
    run_valuation_scenarios, run_price_simulation,
)

st.set_page_config(
    page_title="PINE4 | Equity Research Terminal",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --bg: #0a0e17; --bg-2: #0f1419; --card: #111827; --card-2: #1a2332;
    --border: #1e2d3d; --green: #00d26a; --red: #ff3b3b; --blue: #0ea5e9;
    --gold: #facc15; --pine: #16a34a; --text: #e2e8f0; --text-dim: #94a3b8; --muted: #64748b;
}
.stApp { background-color: var(--bg) !important; color: var(--text); }
.block-container { padding-top: 1.5rem; max-width: 1500px; padding-bottom: 4rem; }
h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; color: var(--text) !important; letter-spacing: -0.3px; font-weight: 700; }
h1 { font-size: 2rem !important; border-bottom: 2px solid var(--pine); padding-bottom: 0.5rem; margin-bottom: 1.5rem !important; }
h2 { font-size: 1.4rem !important; color: var(--blue) !important; }
h3 { font-size: 1.1rem !important; color: var(--blue) !important; }
p, .stMarkdown { font-family: 'Inter', sans-serif; color: var(--text-dim); line-height: 1.6; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; background: var(--card-2) !important; border: 1px solid var(--border); border-radius: 4px; }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; border-left: 3px solid var(--pine); }
[data-testid="stMetric"] label { font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 24px !important; color: var(--text) !important; font-weight: 700 !important; }
[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--border); }
.stTabs [data-baseweb="tab-list"] { background: var(--card); border-bottom: 1px solid var(--border); gap: 0; }
.stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; font-size: 12px !important; text-transform: uppercase !important; letter-spacing: 1.2px !important; padding: 12px 20px !important; color: var(--text-dim) !important; border-right: 1px solid var(--border); background: transparent !important; }
.stTabs [aria-selected="true"] { color: var(--pine) !important; background: var(--card-2) !important; border-top: 2px solid var(--pine) !important; }
.stButton > button { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; letter-spacing: 1px !important; text-transform: uppercase !important; border-radius: 4px !important; border: none !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--pine), #15803d) !important; color: white !important; }
.stDataFrame { font-family: 'JetBrains Mono', monospace !important; font-size: 11.5px !important; }
.bloomberg-header { background: linear-gradient(135deg, var(--card), var(--card-2)); border: 1px solid var(--border); border-left: 4px solid var(--pine); padding: 16px 20px; border-radius: 6px; margin-bottom: 16px; }
.bloomberg-header .title { font-family: 'Inter', sans-serif; font-size: 18px; font-weight: 700; color: var(--text); text-transform: uppercase; }
.bloomberg-header .subtitle { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
.edu-box { background: var(--card); border-left: 3px solid var(--gold); padding: 14px 18px; border-radius: 4px; margin: 12px 0; font-family: 'Inter', sans-serif; font-size: 13px; color: var(--text-dim); line-height: 1.6; }
.edu-box strong { color: var(--gold); }
.edu-box code { background: var(--bg) !important; color: var(--blue) !important; padding: 1px 6px; border-radius: 3px; font-size: 11px; }
.thesis-box { background: var(--card); border-left: 3px solid var(--pine); padding: 16px 20px; border-radius: 4px; margin: 12px 0; font-family: 'Inter', sans-serif; font-size: 13px; color: var(--text-dim); line-height: 1.7; }
.thesis-box strong { color: var(--pine); }
.rec-buy { background: rgba(0, 210, 106, 0.15); border-left: 4px solid var(--pine); padding: 14px 18px; border-radius: 4px; }
.rec-hold { background: rgba(250, 204, 21, 0.15); border-left: 4px solid var(--gold); padding: 14px 18px; border-radius: 4px; }
.rec-sell { background: rgba(255, 59, 59, 0.15); border-left: 4px solid var(--red); padding: 14px 18px; border-radius: 4px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PL = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#111827",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#e2e8f0"),
    margin=dict(l=50, r=20, t=60, b=50),
    title_font=dict(family="Inter, sans-serif", size=14, color="#e2e8f0"),
    xaxis=dict(gridcolor="#1e2d3d", zerolinecolor="#1e2d3d"),
    yaxis=dict(gridcolor="#1e2d3d", zerolinecolor="#1e2d3d"),
)

# ============================================================
# HELPERS
# ============================================================
def fmt_t(t):
    return str(t).replace(".SA", "")

def fmt_brl(v, dec=2):
    if pd.isna(v): return "-"
    return "R$ {:,.{}f}".format(v, dec)

def fmt_x(v):
    if pd.isna(v): return "-"
    return "{:.1f}x".format(v)

def fmt_pct(v):
    if pd.isna(v): return "-"
    return "{:.1f}%".format(v * 100)

def fmt_pct_signed(v):
    if pd.isna(v): return "-"
    return "{:+.1f}%".format(v * 100)

def fmt_mcap(v):
    if pd.isna(v): return "-"
    if v >= 1e12: return "R$ {:.2f} T".format(v / 1e12)
    if v >= 1e9: return "R$ {:.2f} B".format(v / 1e9)
    if v >= 1e6: return "R$ {:.1f} M".format(v / 1e6)
    return "R$ {:,.0f}".format(v)

def bloomberg_header(title, subtitle):
    html = '<div class="bloomberg-header"><div class="title">' + title + '</div><div class="subtitle">' + subtitle + '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

def edu_box(html_content):
    st.markdown('<div class="edu-box">' + html_content + '</div>', unsafe_allow_html=True)

def thesis_box(html_content):
    st.markdown('<div class="thesis-box">' + html_content + '</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("# PINE4")
    st.markdown("### EQUITY RESEARCH TERMINAL")
    st.caption("Banco Pine | FGV 2026")
    st.markdown("---")
    st.markdown("##### PEER GROUP")
    peer_lines = []
    for t, (n, _) in PEER_GROUP.items():
        peer_lines.append("`" + fmt_t(t) + "` " + n)
    st.markdown("  \n".join(peer_lines))
    st.markdown("---")
    st.markdown("##### LOOKBACK")
    data_inicio = st.date_input(
        "Start",
        value=pd.Timestamp("2020-01-01"),
        label_visibility="collapsed",
    )
    st.markdown("---")
    run_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("**Engine:** RF Walk-Forward + Multi-Factor")
    st.caption("**Monte Carlo:** GBM Price Simulation")
    st.caption("**Statements:** Annual + Quarterly")

# ============================================================
# STATE
# ============================================================
state_keys = [
    "prices", "fund_df", "statements", "kpis",
    "ml_metrics", "ml_fi", "ml_preds",
    "trading_curves", "trading_summary",
    "peer_comp", "valuation_scen",
    "sim_paths", "sim_stats",
]
for k in state_keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ============================================================
# RUN ANALYSIS
# ============================================================
if run_btn:
    all_tickers = list(PEER_GROUP.keys())
    progress = st.progress(0, "INITIALIZING PINE BANK ANALYSIS...")
    try:
        progress.progress(10, "FETCHING PRICE DATA (Pine + peers)...")
        st.session_state.prices = fetch_prices(all_tickers, str(data_inicio))

        progress.progress(25, "FETCHING BANK FUNDAMENTALS (4-layer fallback)...")
        st.session_state.fund_df = fetch_bank_fundamentals(all_tickers, st.session_state.prices)

        progress.progress(40, "FETCHING FINANCIAL STATEMENTS (Pine)...")
        st.session_state.statements = fetch_financial_statements(PINE_TICKER)
        st.session_state.kpis = compute_bank_kpis(st.session_state.statements)

        progress.progress(55, "PEER COMPARISON RANKING...")
        st.session_state.peer_comp = run_peer_comparison(st.session_state.fund_df)

        progress.progress(65, "VALUATION SCENARIOS (peer multiples)...")
        st.session_state.valuation_scen = run_valuation_scenarios(None, st.session_state.fund_df)

        progress.progress(75, "TRAINING ML MODEL...")
        ml = run_ml(st.session_state.prices, st.session_state.fund_df)
        st.session_state.ml_metrics, st.session_state.ml_fi, st.session_state.ml_preds = ml

        progress.progress(85, "BACKTESTING TRADING SYSTEM...")
        tc, ts = run_trading_system(st.session_state.prices)
        st.session_state.trading_curves = tc
        st.session_state.trading_summary = ts

        progress.progress(95, "MONTE CARLO PRICE SIMULATION...")
        paths, stats = run_price_simulation(st.session_state.prices, PINE_TICKER)
        st.session_state.sim_paths = paths
        st.session_state.sim_stats = stats

        progress.progress(100, "ANALYSIS COMPLETE")
    except Exception as e:
        st.error("ERROR: " + str(e))
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ============================================================
# WELCOME SCREEN
# ============================================================
if st.session_state.prices is None:
    st.markdown("# PINE4 - Banco Pine")
    st.markdown("### Equity Research Terminal | Single-Asset Deep Dive")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
**ABOUT BANCO PINE**

Middle-market commercial bank specialized in corporate credit for medium-sized companies
(R$50M-R$1B revenue). Listed on B3 since 2007. Focus on secured lending, FX & derivatives,
and structured finance.
        """)
    with col_b:
        st.markdown("""
**ENGINE**

8 specialized tools: prices, bank fundamentals, financial statements, bank KPIs,
ML walk-forward, trading system, peer ranking, valuation scenarios, Monte Carlo.
        """)
    with col_c:
        st.markdown("""
**OUTPUT**

Full equity research dossier: thesis, fundamentals, peer comparison, ML forecasts,
valuation scenarios (Bear/Base/Bull), and 1-year Monte Carlo price simulation.
        """)

    st.markdown("---")
    st.markdown("##### GETTING STARTED")
    st.markdown("Click **RUN ANALYSIS** in the sidebar to execute the full pipeline.")

    st.markdown("---")
    edu_box("""
    <strong>ABOUT THIS TERMINAL</strong><br><br>
    This is a <strong>single-asset deep dive</strong> for Banco Pine (PINE4), comparing it against
    a curated peer group of 8 Brazilian banks: ABCB4 (ABC Brasil), BPAC11 (BTG Pactual),
    BMGB4 (BMG), BRSR6 (Banrisul), and the big 4 universal banks (ITUB4, BBDC4, BBAS3, SANB11).<br><br>
    Unlike multi-asset screeners, this terminal provides <strong>institutional-grade equity research</strong>:
    full income statement / balance sheet / cashflow analysis, banking-specific KPIs (ROE, ROA, NIM proxy,
    leverage ratio), valuation scenarios using peer multiples, and Monte Carlo Geometric Brownian Motion
    price simulation for the next 12 months.<br><br>
    <strong>Project:</strong> FGV - IA Aplicada ao Mercado Financeiro | Banco Pine Custom Edition
    """)
    st.stop()

# ============================================================
# DATA SHORTCUTS
# ============================================================
prices = st.session_state.prices
fund_df = st.session_state.fund_df
statements = st.session_state.statements
kpis = st.session_state.kpis
peer_comp = st.session_state.peer_comp
val_scen = st.session_state.valuation_scen
ml_metrics = st.session_state.ml_metrics
ml_fi = st.session_state.ml_fi
ml_preds = st.session_state.ml_preds
trading_curves = st.session_state.trading_curves
trading_summary = st.session_state.trading_summary
sim_paths = st.session_state.sim_paths
sim_stats = st.session_state.sim_stats

pine_row = fund_df[fund_df["ticker"] == PINE_TICKER].iloc[0] if (
    PINE_TICKER in fund_df["ticker"].values
) else None

# ============================================================
# HEADER
# ============================================================
st.markdown("# PINE4 - Banco Pine S.A.")
if pine_row is not None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("PRICE", fmt_brl(pine_row.get("preco")))
    c2.metric("MKT CAP", fmt_mcap(pine_row.get("marketCap")))
    c3.metric("P/E", fmt_x(pine_row.get("P_L")))
    c4.metric("P/B", fmt_x(pine_row.get("P_VP")))
    c5.metric("ROE", fmt_pct(pine_row.get("returnOnEquity")))
    c6.metric("DIV YIELD", fmt_pct(pine_row.get("div_yield")))

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "THESIS", "FUNDAMENTALS", "FINANCIALS", "PEERS",
    "VALUATION", "ML & SIGNALS", "MONTE CARLO"
])

# ----- TAB 1: THESIS -----
with tab1:
    bloomberg_header("INVESTMENT THESIS", "PINE4 | EXECUTIVE SUMMARY")
    if pine_row is not None:
        thesis_box("""
        <strong>BANCO PINE (PINE4) - SNAPSHOT</strong><br><br>
        Banco Pine e um banco brasileiro de <strong>middle-market</strong> focado em credito corporativo
        para empresas de medio porte. Listado na B3 desde 2007, o Pine se diferencia pela atuacao em
        <strong>operacoes estruturadas, FX, derivativos e credito com garantia real</strong>.<br><br>
        <strong>Diferenciacao vs peers:</strong><br>
        - Foco em <strong>middle market</strong> (nao compete com varejo dos grandes bancos)<br>
        - Modelo de negocios <strong>relacional</strong> com base de clientes corporativos<br>
        - Estrutura de capital mais leve que big banks (menor leverage)<br>
        - Maior <strong>spread bancario</strong> por atuar em segmento com menos concorrencia
        """)

        st.markdown("##### KEY METRICS vs PEER AVERAGE")
        peers_only = fund_df[fund_df["ticker"] != PINE_TICKER]
        cmp_rows = []
        cmp_specs = [
            ("P/E", "P_L", fmt_x, True),
            ("P/B", "P_VP", fmt_x, True),
            ("ROE", "returnOnEquity", fmt_pct, False),
            ("ROA", "returnOnAssets", fmt_pct, False),
            ("Profit Margin", "profitMargins", fmt_pct, False),
            ("Leverage", "leverage_ratio", fmt_x, True),
        ]
        for label, col, fmt_func, lower_is_better in cmp_specs:
            if col in fund_df.columns:
                pine_v = pine_row.get(col)
                peer_avg = peers_only[col].median() if not peers_only.empty else np.nan
                diff = "-"
                if pd.notna(pine_v) and pd.notna(peer_avg):
                    if lower_is_better:
                        diff = "BETTER" if pine_v < peer_avg else "WORSE"
                    else:
                        diff = "BETTER" if pine_v > peer_avg else "WORSE"
                cmp_rows.append({
                    "METRIC": label,
                    "PINE4": fmt_func(pine_v),
                    "PEER MEDIAN": fmt_func(peer_avg),
                    "VS PEERS": diff,
                })
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

        if peer_comp is not None and not peer_comp.empty and "composite_pct" in peer_comp.columns:
            pine_pct_row = peer_comp[peer_comp["ticker"] == PINE_TICKER]
            if not pine_pct_row.empty:
                pct = pine_pct_row.iloc[0]["composite_pct"]
                rank_val = int(pine_pct_row.iloc[0].get("overall_rank", 0))
                total = len(peer_comp)
                if pct >= 60:
                    rec, css, color = "BUY", "rec-buy", "#00d26a"
                elif pct >= 40:
                    rec, css, color = "HOLD", "rec-hold", "#facc15"
                else:
                    rec, css, color = "SELL", "rec-sell", "#ff3b3b"
                rec_html = (
                    '<div class="' + css + '">'
                    '<strong style="font-size:18px;color:' + color + ';">RECOMMENDATION: ' + rec + '</strong><br>'
                    'Pine ranks <strong>#' + str(rank_val) + ' of ' + str(total) + '</strong> in the peer group composite score '
                    '(percentile: <strong>' + "{:.0f}".format(pct) + '</strong>).<br>'
                    'The recommendation aggregates ROE, ROA, P/E, P/B, leverage, profit margin and growth.'
                    '</div>'
                )
                st.markdown(rec_html, unsafe_allow_html=True)

    edu_box("""
    <strong>HOW TO READ THIS THESIS</strong><br>
    Equity research starts with a clear <strong>investment thesis</strong> that explains <em>why</em>
    we should care about this stock. The thesis covers three dimensions: <strong>business model</strong>
    (what makes Pine different), <strong>relative valuation</strong> (cheap vs peers?), and
    <strong>recommendation</strong> (BUY / HOLD / SELL based on quantitative + qualitative analysis).<br><br>
    The recommendation here is <strong>quantitative-driven</strong>: it ranks Pine across 7 banking metrics
    vs an 8-bank peer group and computes a composite percentile. A human analyst would layer qualitative
    factors (management quality, credit cycle outlook, regulatory environment) on top.
    """)

# ----- TAB 2: FUNDAMENTALS -----
with tab2:
    bloomberg_header("BANK FUNDAMENTALS", "PINE4 vs PEER GROUP | YAHOO FINANCE DATA")
    edu_box("""
    <strong>KEY BANKING METRICS EXPLAINED</strong><br>
    <strong>ROE (Return on Equity):</strong> net income / shareholder equity. The single most important
    metric for banks. Top-tier Brazilian banks deliver ROE of 18-22%; underperformers below 12%.<br>
    <strong>ROA (Return on Assets):</strong> net income / total assets. For banks, ROA is typically
    1-2% (much lower than corporates because of leverage). ROA x Leverage = ROE.<br>
    <strong>P/B (Price-to-Book):</strong> the most relevant valuation multiple for banks since
    earnings can be volatile but book value is stable. P/B &lt; 1 often signals deep value or distress.<br>
    <strong>Leverage Ratio:</strong> total assets / equity. Banks naturally operate with 8-15x leverage
    (vs 1-3x for corporates). Higher leverage amplifies both gains and losses.<br>
    <strong>NIM Proxy (Net Interest Margin):</strong> approximated via Revenue / Assets. Higher = banks
    earning more per R$ of assets.
    """)

    st.markdown("##### FUNDAMENTALS - ALL BANKS")
    display_cols = ["ticker", "nome", "segment", "preco", "marketCap",
                    "P_L", "P_VP", "returnOnEquity", "returnOnAssets",
                    "profitMargins", "leverage_ratio", "asset_turnover", "revenueGrowth"]
    avail = [c for c in display_cols if c in fund_df.columns]
    tbl = fund_df[avail].copy()
    tbl["ticker"] = tbl["ticker"].apply(fmt_t)
    if "preco" in tbl.columns:
        tbl["preco"] = tbl["preco"].apply(fmt_brl)
    if "marketCap" in tbl.columns:
        tbl["marketCap"] = tbl["marketCap"].apply(fmt_mcap)
    for c in ["P_L", "P_VP", "leverage_ratio", "asset_turnover"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(fmt_x)
    for c in ["returnOnEquity", "returnOnAssets", "profitMargins", "revenueGrowth"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(fmt_pct)
    rename = {"ticker": "TICKER", "nome": "NAME", "segment": "SEGMENT",
              "preco": "PRICE", "marketCap": "MKT CAP", "P_L": "P/E", "P_VP": "P/B",
              "returnOnEquity": "ROE", "returnOnAssets": "ROA",
              "profitMargins": "MARGIN", "leverage_ratio": "LEV",
              "asset_turnover": "NIM PROXY", "revenueGrowth": "GROWTH"}
    tbl = tbl.rename(columns=rename)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("##### ROE & ROA COMPARISON")
    col1, col2 = st.columns(2)
    with col1:
        d = fund_df.dropna(subset=["returnOnEquity"]).copy()
        d["ticker_short"] = d["ticker"].apply(fmt_t)
        d = d.sort_values("returnOnEquity", ascending=True)
        colors = ["#00d26a" if t == PINE_TICKER else "#0ea5e9" for t in d["ticker"]]
        fig = go.Figure(go.Bar(
            y=d["ticker_short"], x=d["returnOnEquity"] * 100,
            orientation="h", marker_color=colors,
            text=["{:.1f}%".format(v * 100) for v in d["returnOnEquity"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.update_layout(title="ROE (%)", **PL)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        d = fund_df.dropna(subset=["returnOnAssets"]).copy()
        d["ticker_short"] = d["ticker"].apply(fmt_t)
        d = d.sort_values("returnOnAssets", ascending=True)
        colors = ["#00d26a" if t == PINE_TICKER else "#0ea5e9" for t in d["ticker"]]
        fig = go.Figure(go.Bar(
            y=d["ticker_short"], x=d["returnOnAssets"] * 100,
            orientation="h", marker_color=colors,
            text=["{:.2f}%".format(v * 100) for v in d["returnOnAssets"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.update_layout(title="ROA (%)", **PL)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        d = fund_df.dropna(subset=["P_L"]).copy()
        d["ticker_short"] = d["ticker"].apply(fmt_t)
        d = d.sort_values("P_L", ascending=True)
        colors = ["#00d26a" if t == PINE_TICKER else "#0ea5e9" for t in d["ticker"]]
        fig = go.Figure(go.Bar(
            y=d["ticker_short"], x=d["P_L"],
            orientation="h", marker_color=colors,
            text=["{:.1f}x".format(v) for v in d["P_L"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.update_layout(title="P/E RATIO", **PL)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        d = fund_df.dropna(subset=["P_VP"]).copy()
        d["ticker_short"] = d["ticker"].apply(fmt_t)
        d = d.sort_values("P_VP", ascending=True)
        colors = ["#00d26a" if t == PINE_TICKER else "#0ea5e9" for t in d["ticker"]]
        fig = go.Figure(go.Bar(
            y=d["ticker_short"], x=d["P_VP"],
            orientation="h", marker_color=colors,
            text=["{:.2f}x".format(v) for v in d["P_VP"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.update_layout(title="P/B RATIO", **PL)
        st.plotly_chart(fig, use_container_width=True)

# ----- TAB 3: FINANCIALS -----
with tab3:
    bloomberg_header("FINANCIAL STATEMENTS", "PINE4 | INCOME, BALANCE SHEET, CASHFLOW")
    edu_box("""
    <strong>DEMONSTRACOES FINANCEIRAS (DRE, BALANCO, FLUXO DE CAIXA)</strong><br>
    Toda analise de equity research seria comeca pelas tres demonstracoes:<br>
    <strong>DRE (Income Statement):</strong> mostra a "performance" do banco - receitas com juros,
    despesas operacionais, provisoes para devedores duvidosos, lucro liquido.<br>
    <strong>Balance Sheet:</strong> a "fotografia" do banco em uma data especifica - ativos
    (carteira de credito, TVM), passivos (depositos, divida), patrimonio liquido.<br>
    <strong>Cashflow:</strong> rastreia o caixa de verdade - operacional, investimentos, financiamento.
    Os dados abaixo vem direto do Yahoo Finance via <code>yf.Ticker('PINE4.SA').financials</code>.
    """)

    if kpis is not None and not kpis.empty:
        st.markdown("##### BANKING KPIs - ANNUAL HISTORY")
        kpi_display = kpis.copy()
        for c in ["net_income", "revenue", "operating_income", "total_assets",
                  "total_equity", "total_debt"]:
            if c in kpi_display.columns:
                kpi_display[c] = kpi_display[c].apply(fmt_mcap)
        for c in ["ROE", "ROA", "net_margin", "NIM_proxy"]:
            if c in kpi_display.columns:
                kpi_display[c] = kpi_display[c].apply(fmt_pct)
        for c in ["leverage", "debt_to_equity"]:
            if c in kpi_display.columns:
                kpi_display[c] = kpi_display[c].apply(fmt_x)
        kpi_display = kpi_display.rename(columns={
            "year": "YEAR", "net_income": "NET INC.", "revenue": "REVENUE",
            "operating_income": "OP. INC.", "total_assets": "ASSETS",
            "total_equity": "EQUITY", "total_debt": "DEBT",
            "ROE": "ROE", "ROA": "ROA", "net_margin": "MARGIN",
            "leverage": "LEV.", "debt_to_equity": "D/E", "NIM_proxy": "NIM",
        })
        st.dataframe(kpi_display, use_container_width=True, hide_index=True)

        st.markdown("##### ROE EVOLUTION (PINE4)")
        roe_data = kpis.dropna(subset=["ROE"]).copy()
        if not roe_data.empty:
            fig = go.Figure(go.Bar(
                x=roe_data["year"].astype(str),
                y=roe_data["ROE"] * 100,
                marker_color=["#00d26a" if v > 0 else "#ff3b3b" for v in roe_data["ROE"]],
                text=["{:+.1f}%".format(v * 100) for v in roe_data["ROE"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=12),
            ))
            fig.add_hline(y=15, line_dash="dash", line_color="#facc15", opacity=0.5,
                          annotation_text="Top-tier threshold (15%)")
            fig.update_layout(title="ROE BY YEAR (%)", yaxis_title="ROE (%)", **PL)
            st.plotly_chart(fig, use_container_width=True)

        ni_data = kpis.dropna(subset=["net_income"]).copy()
        if not ni_data.empty:
            fig = go.Figure(go.Bar(
                x=ni_data["year"].astype(str),
                y=ni_data["net_income"] / 1e6,
                marker_color=["#00d26a" if v > 0 else "#ff3b3b" for v in ni_data["net_income"]],
                text=["R${:.0f}M".format(v / 1e6) for v in ni_data["net_income"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=12),
            ))
            fig.update_layout(title="NET INCOME BY YEAR (R$ millions)",
                              yaxis_title="NET INC (R$M)", **PL)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No financial statements data available for PINE4.")

    st.markdown("##### RAW INCOME STATEMENT (Annual)")
    if statements and not statements.get("income", pd.DataFrame()).empty:
        inc = statements["income"].copy()
        inc.columns = [c.year if hasattr(c, "year") else str(c)[:10] for c in inc.columns]
        inc_fmt = inc.applymap(lambda v: fmt_mcap(v) if pd.notna(v) else "-")
        st.dataframe(inc_fmt, use_container_width=True)
    else:
        st.info("Income statement not available.")

    st.markdown("##### RAW BALANCE SHEET (Annual)")
    if statements and not statements.get("balance", pd.DataFrame()).empty:
        bs = statements["balance"].copy()
        bs.columns = [c.year if hasattr(c, "year") else str(c)[:10] for c in bs.columns]
        bs_fmt = bs.applymap(lambda v: fmt_mcap(v) if pd.notna(v) else "-")
        st.dataframe(bs_fmt, use_container_width=True)
    else:
        st.info("Balance sheet not available.")

# ----- TAB 4: PEERS -----
with tab4:
    bloomberg_header("PEER COMPARISON", "PINE4 vs 8 BRAZILIAN BANKS | COMPOSITE RANKING")
    edu_box("""
    <strong>PEER ANALYSIS</strong><br>
    Banks are compared against <strong>relevant peers</strong> - other banks with similar business
    models, not against general indices. Our peer group includes 3 middle-market peers (ABC Brasil,
    BMG, Banrisul), 1 investment bank (BTG), and the big 4 universal banks. The
    <strong>composite percentile</strong> aggregates 7 metrics: ROE, ROA, P/E, P/B, leverage,
    profit margin, and revenue growth.
    """)

    if peer_comp is not None and not peer_comp.empty and "composite_pct" in peer_comp.columns:
        d = peer_comp.dropna(subset=["composite_pct"]).copy()
        d["ticker_short"] = d["ticker"].apply(fmt_t)
        d = d.sort_values("composite_pct", ascending=True)
        colors = ["#00d26a" if t == PINE_TICKER else "#0ea5e9" for t in d["ticker"]]
        fig = go.Figure(go.Bar(
            y=d["ticker_short"], x=d["composite_pct"],
            orientation="h", marker_color=colors,
            text=["{:.0f}".format(v) for v in d["composite_pct"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12),
        ))
        fig.add_vline(x=50, line_dash="dash", line_color="#64748b", opacity=0.5)
        fig.update_layout(title="COMPOSITE PERCENTILE (0-100, higher = better)",
                          xaxis_range=[0, 110], **PL)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### RANK PER METRIC")
        rank_cols = [c for c in peer_comp.columns if c.endswith("_rank")]
        if rank_cols:
            rank_tbl = peer_comp[["ticker"] + rank_cols].copy()
            rank_tbl["ticker"] = rank_tbl["ticker"].apply(fmt_t)
            new_cols = ["TICKER"]
            for c in rank_cols:
                new_cols.append(c.replace("_rank", "").upper())
            rank_tbl.columns = new_cols
            for c in rank_tbl.columns[1:]:
                rank_tbl[c] = rank_tbl[c].apply(lambda v: "#" + str(int(v)) if pd.notna(v) else "-")
            st.dataframe(rank_tbl, use_container_width=True, hide_index=True)

    if pine_row is not None:
        st.markdown("##### PINE4 RADAR (vs peer median)")
        peers_only = fund_df[fund_df["ticker"] != PINE_TICKER]
        radar_metrics = ["returnOnEquity", "returnOnAssets", "profitMargins"]
        avail_radar = [m for m in radar_metrics if m in fund_df.columns]
        if avail_radar:
            categories = []
            for m in avail_radar:
                cat = m.replace("returnOn", "RO").replace("Equity", "E")
                cat = cat.replace("Assets", "A").replace("Margins", "MARGIN").upper()
                categories.append(cat)
            pine_vals = [(pine_row.get(m, 0) * 100) if pd.notna(pine_row.get(m)) else 0
                         for m in avail_radar]
            peer_vals = [(peers_only[m].median() * 100) if pd.notna(peers_only[m].median()) else 0
                         for m in avail_radar]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=pine_vals + [pine_vals[0]],
                theta=categories + [categories[0]],
                name="PINE4", fill="toself", line_color="#00d26a",
            ))
            fig.add_trace(go.Scatterpolar(
                r=peer_vals + [peer_vals[0]],
                theta=categories + [categories[0]],
                name="PEER MEDIAN", fill="toself", line_color="#0ea5e9", opacity=0.6,
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="#111827",
                    radialaxis=dict(visible=True, gridcolor="#1e2d3d"),
                    angularaxis=dict(gridcolor="#1e2d3d"),
                ),
                title="PINE4 vs PEER MEDIAN",
                paper_bgcolor="#0a0e17",
                font=dict(family="JetBrains Mono, monospace", size=11, color="#e2e8f0"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ----- TAB 5: VALUATION -----
with tab5:
    bloomberg_header("VALUATION SCENARIOS", "PINE4 | BEAR / BASE / BULL FROM PEER MULTIPLES")
    edu_box("""
    <strong>VALUATION POR MULTIPLOS DE PARES</strong><br>
    Em equity research, uma abordagem classica e estimar o "preco justo" aplicando multiplos do peer
    group aos fundamentos da empresa-alvo. Geramos <strong>3 cenarios</strong>:<br>
    - <strong>Bear (p25):</strong> peer multiple no percentil 25 (cenario pessimista)<br>
    - <strong>Base (median):</strong> peer multiple na mediana (cenario neutro)<br>
    - <strong>Bull (p75):</strong> peer multiple no percentil 75 (cenario otimista)<br><br>
    O <strong>upside %</strong> mostra a distancia do preco atual ao preco implicito de cada cenario.
    """)

    if val_scen is not None and not val_scen.empty:
        st.markdown("##### VALUATION SCENARIOS TABLE")
        scen_tbl = val_scen.copy()
        scen_tbl["implied_price"] = scen_tbl["implied_price"].apply(
            lambda v: fmt_brl(v) if pd.notna(v) else "-")
        scen_tbl["current_price"] = scen_tbl["current_price"].apply(
            lambda v: fmt_brl(v) if pd.notna(v) else "-")
        scen_tbl["upside_pct"] = scen_tbl["upside_pct"].apply(
            lambda v: "{:+.1f}%".format(v) if pd.notna(v) else "-")
        scen_tbl["peer_multiple"] = scen_tbl["peer_multiple"].apply(
            lambda v: fmt_x(v) if pd.notna(v) else "-")
        scen_tbl.columns = ["MULTIPLE", "SCENARIO", "PEER MULT",
                            "IMPLIED PRICE", "CURRENT PRICE", "UPSIDE %"]
        st.dataframe(scen_tbl, use_container_width=True, hide_index=True)

        st.markdown("##### IMPLIED PRICE vs CURRENT")
        v = val_scen.dropna(subset=["implied_price", "upside_pct"]).copy()
        if not v.empty:
            v["label"] = v["multiple"] + " - " + v["scenario"]
            colors = ["#00d26a" if u > 0 else "#ff3b3b" for u in v["upside_pct"]]
            fig = go.Figure(go.Bar(
                y=v["label"], x=v["upside_pct"],
                orientation="h", marker_color=colors,
                text=["{:+.1f}%".format(u) for u in v["upside_pct"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=12),
            ))
            fig.add_vline(x=0, line_color="#64748b", opacity=0.5)
            fig.update_layout(title="UPSIDE / DOWNSIDE vs CURRENT PRICE (%)",
                              xaxis_title="UPSIDE (%)", **PL)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Valuation scenarios not available - missing financial data.")

# ----- TAB 6: ML & SIGNALS -----
with tab6:
    bloomberg_header("ML & TRADING SIGNALS", "RANDOM FOREST + MA CROSSOVER")
    edu_box("""
    <strong>MULTI-FACTOR RANDOM FOREST</strong><br>
    O modelo combina <strong>features tecnicas</strong> (vol, momentum 3m/6m/12m, drawdown) com
    <strong>features fundamentalistas</strong> (P/E, P/B, ROE, ROA, margem, growth, leverage).
    Todas as features sao <strong>z-scored cross-sectionally</strong> por data. A validacao
    e <strong>walk-forward</strong>: treina em t-N, testa em t.
    """)

    if ml_metrics is not None and not ml_metrics.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### WALK-FORWARD METRICS")
            st.dataframe(ml_metrics.round(4), use_container_width=True, hide_index=True)
            fig = go.Figure(go.Bar(
                x=ml_metrics["year"].astype(str),
                y=ml_metrics["spearman_ic"],
                marker_color=["#00d26a" if v > 0 else "#ff3b3b" for v in ml_metrics["spearman_ic"]],
                text=["{:.3f}".format(v) for v in ml_metrics["spearman_ic"]],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig.add_hline(y=0, line_color="#64748b")
            fig.update_layout(title="SPEARMAN IC BY YEAR", **PL)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if ml_fi is not None and not ml_fi.empty:
                st.markdown("##### FEATURE IMPORTANCE")
                fi_mean = ml_fi.mean().sort_values(ascending=True)
                fig = go.Figure(go.Bar(
                    y=fi_mean.index, x=fi_mean.values,
                    orientation="h", marker_color="#16a34a",
                    text=["{:.3f}".format(v) for v in fi_mean.values],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=10),
                ))
                fig.update_layout(title="AVG IMPORTANCE", **PL)
                st.plotly_chart(fig, use_container_width=True)

    if ml_preds is not None and not ml_preds.empty and "pred_ret_12m" in ml_preds.columns:
        st.markdown("##### 12-MONTH RETURN FORECAST")
        pp = ml_preds[["ticker", "pred_ret_12m", "rank_pred"]].dropna().copy()
        pp["ticker"] = pp["ticker"].apply(fmt_t)
        pp["pred_ret_12m"] = (pp["pred_ret_12m"] * 100).round(1)
        pp = pp.sort_values("rank_pred")
        colors = []
        for t, v in zip(pp["ticker"], pp["pred_ret_12m"]):
            if t == fmt_t(PINE_TICKER):
                colors.append("#16a34a")
            elif v > 0:
                colors.append("#00d26a")
            else:
                colors.append("#ff3b3b")
        fig = go.Figure(go.Bar(
            x=pp["ticker"], y=pp["pred_ret_12m"],
            marker_color=colors,
            text=["{:+.1f}%".format(v) for v in pp["pred_ret_12m"]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig.add_hline(y=0, line_color="#64748b")
        fig.update_layout(title="PREDICTED 12M RETURN (%)",
                          yaxis_title="RETURN (%)", **PL)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### MA CROSSOVER STRATEGY (PINE4)")
    if trading_curves is not None and not trading_curves.empty:
        pine_curve = trading_curves[trading_curves["ticker"] == PINE_TICKER]
        if not pine_curve.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pine_curve["data"], y=pine_curve["acum_acao"],
                name="BUY & HOLD",
                line=dict(color="#0ea5e9", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=pine_curve["data"], y=pine_curve["acum_est"],
                name="MA STRATEGY",
                line=dict(color="#16a34a", width=2),
            ))
            fig.add_hline(y=1, line_dash="dash", line_color="#64748b", opacity=0.4)
            fig.update_layout(title="PINE4 | EQUITY CURVE | MA(20,60)",
                              yaxis_title="CUMULATIVE WEALTH", **PL)
            st.plotly_chart(fig, use_container_width=True)
            if trading_summary is not None and not trading_summary.empty:
                pine_sum = trading_summary[trading_summary["ticker"] == PINE_TICKER]
                if not pine_sum.empty:
                    s = pine_sum.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("BUY & HOLD", "{:+.1f}%".format(s["ret_buyhold"]))
                    c2.metric("MA STRATEGY", "{:+.1f}%".format(s["ret_estrategia"]))
                    c3.metric("ALPHA", "{:+.1f}%".format(s["alpha"]))

# ----- TAB 7: MONTE CARLO -----
with tab7:
    bloomberg_header("MONTE CARLO SIMULATION", "PINE4 | 1-YEAR GBM PRICE PATHS")
    edu_box("""
    <strong>GEOMETRIC BROWNIAN MOTION (GBM)</strong><br>
    GBM e o modelo padrao de financas quantitativas para simular trajetorias de precos:<br>
    <code>S(t+1) = S(t) * exp[(mu - 1/2*sigma^2)*dt + sigma*sqrt(dt)*Z]</code><br>
    Onde mu e o drift (media de retornos diarios), sigma e a volatilidade, e Z e ruido normal.
    Rodamos <strong>1.000 simulacoes</strong> com horizonte de 252 dias (1 ano de trading).
    """)

    if sim_stats and sim_paths is not None and not sim_paths.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CURRENT PRICE", fmt_brl(sim_stats["current_price"]))
        c2.metric("EXPECTED 1Y", fmt_brl(sim_stats["mean_1y"]),
                  delta="{:+.1f}%".format(sim_stats["expected_return"]))
        c3.metric("P95 (BULL)", fmt_brl(sim_stats["p95_1y"]))
        c4.metric("P5 (BEAR)", fmt_brl(sim_stats["p5_1y"]))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("PROB GAIN", "{:.1f}%".format(sim_stats["prob_gain"]))
        c6.metric("PROB +20%", "{:.1f}%".format(sim_stats["prob_20pct_gain"]))
        c7.metric("PROB -20%", "{:.1f}%".format(sim_stats["prob_20pct_loss"]))
        c8.metric("MEDIAN 1Y", fmt_brl(sim_stats["median_1y"]))

        st.markdown("##### SIMULATED PRICE PATHS (1Y, 100 sample paths)")
        fig = go.Figure()
        for col in sim_paths.columns[:100]:
            fig.add_trace(go.Scatter(
                x=sim_paths.index, y=sim_paths[col],
                mode="lines", line=dict(color="#16a34a", width=0.5),
                opacity=0.3, showlegend=False, hoverinfo="skip",
            ))
        fig.add_hline(y=sim_stats["current_price"], line_dash="dash",
                      line_color="#facc15", opacity=0.8,
                      annotation_text="Current: R${:.2f}".format(sim_stats["current_price"]))
        fig.update_layout(title="MONTE CARLO PATHS - 252 TRADING DAYS",
                          yaxis_title="PRICE (R$)", **PL)
        st.plotly_chart(fig, use_container_width=True)

        final_prices = sim_paths.iloc[-1].values
        st.markdown("##### DISTRIBUTION OF 1-YEAR PRICE")
        fig = go.Figure(go.Histogram(
            x=final_prices, nbinsx=30, marker_color="#16a34a",
            marker_line=dict(color="#0a0e17", width=1),
        ))
        fig.add_vline(x=sim_stats["current_price"], line_dash="dash",
                      line_color="#facc15", annotation_text="Current")
        fig.add_vline(x=sim_stats["p5_1y"], line_dash="dot",
                      line_color="#ff3b3b", annotation_text="P5")
        fig.add_vline(x=sim_stats["p95_1y"], line_dash="dot",
                      line_color="#00d26a", annotation_text="P95")
        fig.update_layout(title="HISTOGRAM OF 1Y SIMULATED PRICES",
                          xaxis_title="PRICE (R$)", yaxis_title="FREQUENCY", **PL)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Monte Carlo simulation not available - insufficient price history.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
footer = (
    '<div style="text-align:center; color:#64748b; font-family:JetBrains Mono; font-size:11px;">'
    'PINE4 EQUITY RESEARCH TERMINAL v1.0 | FGV 2026 | DATA: Yahoo Finance | '
    'LAST UPDATE: ' + datetime.now().strftime("%Y-%m-%d %H:%M") +
    '</div>'
)
st.markdown(footer, unsafe_allow_html=True)
