# 🏦 PINE4 — Banco Pine Equity Research Terminal

**Projeto Final FGV** | IA Aplicada ao Mercado Financeiro | Edição Banco Pine

Terminal de equity research no estilo Bloomberg para análise **single-asset deep dive** do Banco Pine (PINE4), com comparação contra peer group de 8 bancos brasileiros e análise de demonstrações financeiras.

---

## 🚀 Deploy no Streamlit Cloud (5 minutos)

### Passo 1 — Criar novo repositório no GitHub
1. Vá em https://github.com/new
2. Nome: `PINE4-Equity-Research`
3. Marque **Public**
4. **Create repository**

### Passo 2 — Upload dos arquivos
Faça upload de TODOS os arquivos abaixo:
- `app.py`
- `engine.py`
- `requirements.txt`
- `README.md`
- pasta `.streamlit/` com `config.toml` dentro

### Passo 3 — Deploy
1. Vá em https://share.streamlit.io
2. Login com GitHub
3. **New app** → selecione `PINE4-Equity-Research` → Main file: `app.py`
4. **Deploy!**

URL pública pronta em ~2 minutos.

---

## 🏗️ Arquitetura

```
PEER GROUP (9 bancos):
   PINE4 (target) + ABCB4, BPAC11, BMGB4, BRSR6 (middle market + IB)
   + ITUB4, BBDC4, BBAS3, SANB11 (big 4)
   │
   ├─ TOOL 1: fetch_prices                  → Preços + vol, momentum 3/6/12m, drawdown, SMA 20/60/200
   ├─ TOOL 2: fetch_bank_fundamentals       → P/E, P/B, ROE, ROA, leverage, NIM proxy (4-layer fallback)
   ├─ TOOL 3: fetch_financial_statements    → DRE, Balanço, Cashflow anual e trimestral
   ├─ TOOL 4: compute_bank_kpis             → ROE, ROA, leverage, debt/equity, margem ao longo dos anos
   ├─ TOOL 5: run_ml                        → RF walk-forward multi-fator (técnicas + fundamentais)
   ├─ TOOL 6: run_trading_system            → MA Crossover 20/60 com backtest
   ├─ TOOL 7: run_peer_comparison           → Ranking composto Pine vs peers (7 métricas)
   ├─ TOOL 8: run_valuation_scenarios       → Preço justo via Bear/Base/Bull peer multiples
   └─ TOOL 9: run_price_simulation          → Monte Carlo GBM 1.000 simulações 1 ano
```

## 📊 7 Abas do Terminal

| Aba | Conteúdo |
|-----|----------|
| **THESIS** | Tese de investimento, métricas vs peer median, recomendação BUY/HOLD/SELL |
| **FUNDAMENTALS** | Todos os múltiplos do peer group + comparação visual ROE/ROA/P/E/P/B |
| **FINANCIALS** | KPIs bancários ao longo dos anos + DRE e Balanço brutos do Yahoo Finance |
| **PEERS** | Ranking composto percentil + radar chart Pine vs peer median |
| **VALUATION** | Cenários Bear/Base/Bull usando peer multiples (P/E e P/B) |
| **ML & SIGNALS** | RF walk-forward, feature importance, previsão 12m, MA crossover |
| **MONTE CARLO** | Simulação GBM 1.000 caminhos, distribuição P5/P95, probabilidades |

## 🎯 Diferenciais vs Versão Geral

| Aspecto | Versão Geral (B3) | Versão Pine |
|---------|---------------|-------------|
| Foco | Multi-asset (79 ações) | Single-asset deep dive (PINE4) |
| Peer Group | Setores genéricos | 8 bancos curados (middle market + big 4) |
| Métricas | Padrão (vol, mom, P/E) | Bancárias (ROE, ROA, leverage, NIM proxy) |
| Demonstrações | Não | DRE + Balanço + Cashflow + KPIs anuais |
| Valuation | Score composto | Bear/Base/Bull com peer multiples |
| Simulação | Markowitz (carteira) | Monte Carlo GBM (preço individual) |

## 🎓 Conteúdo Didático

Cada aba contém uma caixa amarela 📚 explicando os conceitos:
- ROE, ROA, P/B para bancos
- Leverage ratio e NIM proxy
- Walk-forward validation
- GBM (Geometric Brownian Motion)
- Valuation por múltiplos de pares

## 📁 Estrutura Final

```
PINE4-Equity-Research/
├── app.py                     ← Terminal Streamlit (7 abas)
├── engine.py                  ← 9 tools especializadas em bancos
├── requirements.txt           ← Dependências
├── .streamlit/
│   └── config.toml            ← Dark theme verde Pine
└── README.md                  ← Este arquivo
```

---

*FGV 2026 | IA Aplicada ao Mercado Financeiro | Prof. João Luiz Chela*
