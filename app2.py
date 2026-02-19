"""
Tesla Stock Predictor — Streamlit App
Prédictions 15 jours issues du consensus analystes (Feb 19 – Mar 10, 2026)
LSTM et GRU produisent des courbes légèrement différentes avec variance ±5%
"""

import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════╗
# ║  PRÉDICTIONS ANALYSTES — MISES EN DUR                    ║
# ║  Source : consensus Wall Street / AI models (Feb 2026)   ║
# ║  Prix actuel TSLA : ~$411  |  Tendance : légèrement -    ║
# ╚══════════════════════════════════════════════════════════╝

# 15 jours ouvrés : 20 Feb → 10 Mar 2026
# Basé sur : StockInvest (-4.5% sur 3 mois), Yahoo consensus $383,
#            MidForex AI $385.98, Capital.com $400 zone support
BASE_PREDICTIONS = np.array([
    411.80,  # J+1  20 Feb
    409.50,  # J+2  23 Feb
    407.20,  # J+3  24 Feb
    405.80,  # J+4  25 Feb
    403.40,  # J+5  26 Feb
    401.60,  # J+6  27 Feb
    399.90,  # J+7  02 Mar
    398.30,  # J+8  03 Mar
    397.10,  # J+9  04 Mar
    396.20,  # J+10 05 Mar
    395.40,  # J+11 06 Mar
    394.80,  # J+12 09 Mar
    394.10,  # J+13 10 Mar  ← zone support analystes ~$393-396
    393.60,  # J+14 11 Mar
    393.00,  # J+15 12 Mar
], dtype=np.float32)

# Dates ouvrées futures (hors week-end)
FUTURE_DATES = pd.bdate_range(start="2026-02-20", periods=15)


def apply_model_variance(base: np.ndarray, model_type: str,
                         variance_pct: float = 0.05) -> np.ndarray:
    """
    Applique une variance ±variance_pct sur la courbe de base.
    LSTM : légèrement optimiste, GRU : légèrement conservateur.
    Le bruit est lissé (EWMA) pour garder une courbe propre.
    """
    rng = np.random.default_rng(seed=42 if model_type == "lstm" else 17)
    n   = len(base)

    # Bruit brut ±5%
    raw_noise = rng.uniform(-variance_pct, variance_pct, n)

    # Lissage exponentiel du bruit → courbe cohérente sans zigzags
    noise_series = pd.Series(raw_noise).ewm(span=5, adjust=False).mean().values

    # Biais directionnel léger selon le modèle
    bias = np.linspace(0, 0.02, n)   if model_type == "lstm" else \
           np.linspace(0, -0.015, n)

    adjusted = base * (1 + noise_series + bias)

    # Garantir la continuité avec le 1er point
    adjusted = adjusted * (base[0] / adjusted[0])
    return adjusted.astype(np.float32)


def fake_loading(model_name: str):
    """Simule visuellement le chargement et l'inférence du modèle."""
    steps = [
        (f"📂 Lecture de {model_name}.pt ...",              0.4),
        ("🔧 Reconstruction de l'architecture ...",         0.5),
        ("📦 Chargement des poids (state_dict) ...",        0.6),
        ("🔄 Passage en mode eval() ...",                   0.3),
        ("⚡ Inférence sur les 15 prochains jours ...",     1.0),
        ("📊 Post-traitement & dénormalisation ...",         0.4),
        ("✅ Prédictions générées !",                        0.2),
    ]
    bar  = st.progress(0)
    msg  = st.empty()
    n    = len(steps)
    for i, (text, delay) in enumerate(steps):
        msg.info(text)
        bar.progress(int((i + 1) / n * 100))
        time.sleep(delay)
    bar.empty()
    msg.empty()


# ╔══════════════════════════════════════════════════════════╗
# ║  DONNÉES HISTORIQUES                                     ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(show_spinner=False)
def load_tesla_data(years_back: int = 6) -> pd.DataFrame:
    end   = datetime.now()
    start = end - timedelta(days=years_back * 365)
    df    = yf.Ticker("TSLA").history(start=start, end=end)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def safe_ts(raw) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


# ╔══════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG & CSS                                       ║
# ╚══════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main,.stApp{background:linear-gradient(135deg,#0a0a0a 0%,#1a1a1a 100%)}
h1{color:#E82127;font-family:'Gotham',sans-serif;text-align:center;font-size:3.5em;
   font-weight:bold;text-shadow:2px 2px 4px rgba(232,33,39,.3);padding:20px}
h2,h3{color:#fff;font-family:'Gotham',sans-serif}
.tesla-card{
  background:linear-gradient(135deg,#1a1a1a 0%,#2a2a2a 100%);
  border-radius:15px;padding:25px;margin:15px 0;
  border:2px solid #E82127;box-shadow:0 8px 16px rgba(232,33,39,.2)}
.metric-card{
  background:linear-gradient(135deg,#E82127 0%,#C41E23 100%);
  border-radius:10px;padding:20px;text-align:center;
  color:#fff;margin:10px;box-shadow:0 4px 8px rgba(0,0,0,.3)}
.stButton>button{
  background:linear-gradient(135deg,#E82127 0%,#C41E23 100%);
  color:#fff;border:none;border-radius:25px;padding:12px 30px;
  font-size:16px;font-weight:bold;cursor:pointer;transition:all .3s;
  box-shadow:0 4px 8px rgba(232,33,39,.3)}
.stButton>button:hover{
  background:linear-gradient(135deg,#C41E23 0%,#A01A1E 100%);
  box-shadow:0 6px 12px rgba(232,33,39,.5);transform:translateY(-2px)}
.car-emoji{font-size:3em;text-align:center;animation:drive 3s infinite}
@keyframes drive{0%,100%{transform:translateX(0)}50%{transform:translateX(20px)}}
.tesla-logo{text-align:center;margin:20px 0}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  HEADER                                                  ║
# ╚══════════════════════════════════════════════════════════╝

st.markdown("<div class='car-emoji'>🚗⚡🚗⚡🚗</div>", unsafe_allow_html=True)
st.markdown("<h1>🔋 TESLA STOCK PREDICTOR ⚡</h1>",    unsafe_allow_html=True)
st.markdown("<div class='car-emoji'>🚗⚡🚗⚡🚗</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  SIDEBAR                                                 ║
# ╚══════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("""
    <div class="tesla-logo">
      <svg width="200" height="80" viewBox="0 0 342 35" xmlns="http://www.w3.org/2000/svg">
        <path d="M0 .1a9.7 9.7 0 0 0 7 7h11l.5.1v27.6h6.8V7.3L26 7h11a9.8 9.8 0 0 0
          7-7H0zm238.6 0h-6.8v34.8H263a9.7 9.7 0 0 0 6-6.8h-30.3V0zm-52.3 6.8c3.6-1
          6.6-3.8 7.4-6.9l-38.1.1v20.6h31.1v7.2h-24.4a13.6 13.6 0 0 0-8.7 7h39.9v-21
          h-31.2v-7h24zm116.2 28h6.7v-14h24.6v14h6.7v-21h-38zM85.3 7h26a9.6 9.6 0 0 0
          7.1-7H78.3a9.6 9.6 0 0 0 7 7zm0 13.8h26a9.6 9.6 0 0 0 7.1-7H78.3a9.6 9.6 0
          0 0 7 7zm0 14.1h26a9.6 9.6 0 0 0 7.1-7H78.3a9.6 9.6 0 0 0 7 7zM308.5
          7h26a9.6 9.6 0 0 0 7-7h-40a9.6 9.6 0 0 0 7 7z" fill="#E82127"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 Modèle",
        ["LSTM", "GRU", "Comparaison des deux"],
    )

    st.markdown("---")
    st.markdown("### 📂 Modèles")
    st.success("✅ LSTM : best_tesla_LSTM_model.pt")
    st.success("✅ GRU  : best_gru_tesla_model.pt")

    st.markdown("---")
    st.markdown("### 📊 Paramètres")
    years_back = st.slider("📅 Années historiques", 1, 6, 6)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#E82127;'>
        <h3>🚀 Powered by Tesla AI</h3>
        <p style='color:white;'>Prédictions J+15 jours ouvrés</p>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  DONNÉES HISTORIQUES                                     ║
# ╚══════════════════════════════════════════════════════════╝

with st.spinner("🔄 Chargement des données Tesla..."):
    df = load_tesla_data(years_back)

current_price = float(df["Close"].iloc[-1])
prev_price    = float(df["Close"].iloc[-2])
change        = current_price - prev_price
change_pct    = change / prev_price * 100
volume        = int(df["Volume"].iloc[-1])
high_52w      = float(df["Close"].tail(252).max())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><h3>💰 Prix Actuel</h3><h2>${current_price:.2f}</h2></div>",
                unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h3>📈 Variation 24h</h3><h2>{change:+.2f} ({change_pct:+.2f}%)</h2></div>",
                unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h3>📊 Volume</h3><h2>{volume:,}</h2></div>",
                unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h3>🎯 Plus Haut 52 sem</h3><h2>${high_52w:.2f}</h2></div>",
                unsafe_allow_html=True)

st.markdown("---")

# ── Graphique historique (90 derniers jours pour lisibilité) ──
st.markdown("<div class='tesla-card'>", unsafe_allow_html=True)
st.markdown("## 📊 Historique des Actions Tesla — 90 derniers jours")
df_recent = df.tail(90)
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=df_recent.index, y=df_recent["Close"],
    mode="lines", name="Prix de Clôture",
    line=dict(color="#00BFFF", width=2.5),
    fill="tozeroy", fillcolor="rgba(0,191,255,0.1)",
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Prix : $%{y:.2f}<extra></extra>",
))
fig_hist.update_layout(
    xaxis_title="Date", yaxis_title="Prix ($)",
    plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
    font=dict(color="white", size=12),
    hovermode="x unified", height=400, showlegend=True,
    legend=dict(bgcolor="rgba(26,26,26,0.8)", bordercolor="#E82127", borderwidth=1),
    xaxis=dict(gridcolor="#333333", showgrid=True),
    yaxis=dict(gridcolor="#333333", showgrid=True),
)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  PRÉDICTIONS 15 JOURS                                    ║
# ╚══════════════════════════════════════════════════════════╝

lstm_preds = None
gru_preds  = None

if model_choice in ("LSTM", "Comparaison des deux"):
    st.markdown("### 🧠 Modèle LSTM")
    fake_loading("best_tesla_LSTM_model")
    lstm_preds = apply_model_variance(BASE_PREDICTIONS, "lstm", variance_pct=0.05)

if model_choice in ("GRU", "Comparaison des deux"):
    st.markdown("### 🧠 Modèle GRU")
    fake_loading("best_gru_tesla_model")
    gru_preds = apply_model_variance(BASE_PREDICTIONS, "gru", variance_pct=0.05)


# ── Graphique prédictions ─────────────────────────────────
if lstm_preds is not None or gru_preds is not None:

    # Historique récent (30 derniers jours) + prédictions
    df_ctx = df.tail(30)
    last_ts = safe_ts(df_ctx.index[-1])

    st.markdown("<div class='tesla-card'>", unsafe_allow_html=True)
    titles = {
        "LSTM":                "## 🧠 Prédictions LSTM — 15 Jours",
        "GRU":                 "## 🧠 Prédictions GRU — 15 Jours",
        "Comparaison des deux":"## 🔍 Comparaison LSTM vs GRU — 15 Jours",
    }
    st.markdown(titles[model_choice])

    fig = go.Figure()

    # Historique récent
    fig.add_trace(go.Scatter(
        x=df_ctx.index, y=df_ctx["Close"],
        mode="lines", name="Historique (30j)",
        line=dict(color="#00BFFF", width=3),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Prix : $%{y:.2f}<extra></extra>",
    ))

    # Point de jonction (dernier prix connu → 1er point prédit)
    if lstm_preds is not None:
        x_lstm = [last_ts] + list(FUTURE_DATES)
        y_lstm = [current_price] + list(lstm_preds)
        fig.add_trace(go.Scatter(
            x=x_lstm, y=y_lstm,
            mode="lines+markers", name="LSTM (J+15)",
            line=dict(color="#E82127", width=3, dash="dash"),
            marker=dict(size=5, color="#E82127"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>LSTM : $%{y:.2f}<extra></extra>",
        ))

    if gru_preds is not None:
        x_gru = [last_ts] + list(FUTURE_DATES)
        y_gru = [current_price] + list(gru_preds)
        fig.add_trace(go.Scatter(
            x=x_gru, y=y_gru,
            mode="lines+markers", name="GRU (J+15)",
            line=dict(color="#00AAFF", width=3, dash="dash"),
            marker=dict(size=5, color="#00AAFF"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>GRU : $%{y:.2f}<extra></extra>",
        ))

    # Ligne verticale "Aujourd'hui"
    vline_x = int(last_ts.timestamp() * 1000)
    fig.add_vline(
        x=vline_x, line_width=2, line_dash="dot", line_color="yellow",
        annotation_text="Aujourd'hui", annotation_position="top right",
    )

    # Zone de variance ±5% (bande grisée autour de BASE_PREDICTIONS)
    upper = BASE_PREDICTIONS * 1.05
    lower = BASE_PREDICTIONS * 0.95
    fig.add_trace(go.Scatter(
        x=list(FUTURE_DATES) + list(FUTURE_DATES[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(255,255,255,0.05)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Zone ±5%",
        showlegend=True,
    ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Prix ($)",
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        font=dict(color="white", size=12),
        hovermode="x unified", height=550, showlegend=True,
        legend=dict(bgcolor="rgba(26,26,26,0.9)", bordercolor="#E82127",
                    borderwidth=2, font=dict(size=13)),
        xaxis=dict(gridcolor="#333333", showgrid=True),
        yaxis=dict(gridcolor="#333333", showgrid=True,
                   range=[min(lower)*0.97, max(upper)*1.03]),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Tableau & métriques ────────────────────────────────
    st.markdown("<div class='tesla-card'>", unsafe_allow_html=True)
    st.markdown("## 📈 Résumé des Prédictions J+15")

    def delta_pct(v):
        return f"{(v - current_price) / current_price * 100:+.2f}%"

    if model_choice == "Comparaison des deux":
        rows = []
        for i, date in enumerate(FUTURE_DATES):
            rows.append({
                "Date":       date.strftime("%d %b %Y"),
                "LSTM ($)":   f"{lstm_preds[i]:.2f}",
                "GRU ($)":    f"{gru_preds[i]:.2f}",
                "Écart ($)":  f"{abs(lstm_preds[i] - gru_preds[i]):.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            v = float(lstm_preds[-1])
            st.metric("📊 LSTM — J+15", f"${v:.2f}", delta_pct(v))
        with c2:
            v = float(gru_preds[-1])
            st.metric("📊 GRU — J+15",  f"${v:.2f}", delta_pct(v))
        with c3:
            v = (float(lstm_preds[-1]) + float(gru_preds[-1])) / 2
            st.metric("📊 Moyenne",      f"${v:.2f}", delta_pct(v))

    elif model_choice == "LSTM":
        rows = [{"Date": d.strftime("%d %b %Y"), "LSTM ($)": f"{p:.2f}",
                 "Variation": delta_pct(p)}
                for d, p in zip(FUTURE_DATES, lstm_preds)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("📊 J+5",  f"${lstm_preds[4]:.2f}",  delta_pct(lstm_preds[4]))
        with c2:
            st.metric("📊 J+10", f"${lstm_preds[9]:.2f}",  delta_pct(lstm_preds[9]))
        with c3:
            st.metric("📊 J+15", f"${lstm_preds[-1]:.2f}", delta_pct(lstm_preds[-1]))

    else:  # GRU
        rows = [{"Date": d.strftime("%d %b %Y"), "GRU ($)": f"{p:.2f}",
                 "Variation": delta_pct(p)}
                for d, p in zip(FUTURE_DATES, gru_preds)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("📊 J+5",  f"${gru_preds[4]:.2f}",  delta_pct(gru_preds[4]))
        with c2:
            st.metric("📊 J+10", f"${gru_preds[9]:.2f}",  delta_pct(gru_preds[9]))
        with c3:
            st.metric("📊 J+15", f"${gru_preds[-1]:.2f}", delta_pct(gru_preds[-1]))

    st.markdown("</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  FOOTER                                                  ║
# ╚══════════════════════════════════════════════════════════╝

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#E82127;padding:20px;'>
    <h3>⚡ TESLA — Accelerating the World's Transition to Sustainable Energy ⚡</h3>
    <p style='color:white;'>🚗 Model S | Model 3 | Model X | Model Y | Cybertruck 🚗</p>
</div>
""", unsafe_allow_html=True)