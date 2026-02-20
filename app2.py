"""
Tesla Stock Predictor — Streamlit App
Prédictions 15 jours issues de vos modèles PyTorch (LSTM et GRU)
"""

import time
import warnings
import os
import collections
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════╗
# ║  DÉFINITION DES ARCHITECTURES DE VOS MODÈLES PYTORCH     ║
# ╚══════════════════════════════════════════════════════════╝

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # FIX: only use last timestep output
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # FIX: only use last timestep output
        return out


# ╔══════════════════════════════════════════════════════════╗
# ║  FONCTIONS DE CHARGEMENT ET PRÉDICTION RÉELLES           ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_resource
def load_pytorch_model(model_path, model_type):
    """Charge le modèle PyTorch."""
    if not os.path.exists(model_path):
        return None

    try:
        model_data = torch.load(model_path, map_location=torch.device('cpu'))

        if isinstance(model_data, (dict, collections.OrderedDict)):
            if model_type == "LSTM":
                model = LSTMModel()
            else:
                model = GRUModel()
            model.load_state_dict(model_data)
        else:
            model = model_data

        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {model_type}: {e}")
        return None


def predict_future_real(model, recent_data_scaled, lookback=60, days_to_predict=15):
    """Effectue des prédictions jour par jour (fenêtre glissante autorégressive)."""
    predictions = []
    # FIX: ensure we use exactly 'lookback' days
    current_seq = recent_data_scaled[-lookback:].copy()

    with torch.no_grad():
        for _ in range(days_to_predict):
            # shape (1, lookback, 1)
            x_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred = model(x_tensor).item()
            predictions.append(pred)
            # Sliding window: drop oldest, append new prediction
            current_seq = np.append(current_seq[1:], pred)

    return np.array(predictions)


# ╔══════════════════════════════════════════════════════════╗
# ║  DONNÉES HISTORIQUES (AVEC ANTI RATE-LIMIT YAHOO)        ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(show_spinner=False, ttl=3600)
def load_tesla_data(years_back: int = 6) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years_back * 365)

    try:
        # FIX: Ne jamais passer de session custom à yfinance — il gère curl_cffi en interne
        df = yf.download("TSLA", start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("DataFrame vide retourné par yfinance.")
        # Aplatir les colonnes MultiIndex si nécessaire (yfinance >= 0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Supprimer le timezone de l'index
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Erreur de téléchargement: {e}")
        return pd.DataFrame()


def safe_ts(raw) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


# ╔══════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG & CSS                                       ║
# ╚══════════════════════════════════════════════════════════╝

st.set_page_config(page_title="Tesla AI Predictor", page_icon="🚗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
.main,.stApp{background:linear-gradient(135deg,#0a0a0a 0%,#1a1a1a 100%)}
h1{color:#E82127;font-family:'Rajdhani',sans-serif;text-align:center;font-size:3.5em;font-weight:700;
   text-shadow:0 0 20px rgba(232,33,39,.5),2px 2px 4px rgba(232,33,39,.3);padding:20px}
h2,h3{color:#fff;font-family:'Rajdhani',sans-serif}
.tesla-card{background:linear-gradient(135deg,#1a1a1a 0%,#2a2a2a 100%);border-radius:15px;
   padding:25px;margin:15px 0;border:2px solid #E82127;box-shadow:0 8px 16px rgba(232,33,39,.2)}
.metric-card{background:linear-gradient(135deg,#E82127 0%,#C41E23 100%);border-radius:10px;
   padding:20px;text-align:center;color:#fff;margin:10px;box-shadow:0 4px 8px rgba(0,0,0,.3)}
.metric-card h3{font-family:'Rajdhani',sans-serif;font-size:0.9em;opacity:0.85;margin:0 0 8px 0}
.metric-card h2{font-family:'Share Tech Mono',monospace;font-size:1.4em;margin:0}
.car-emoji{font-size:3em;text-align:center;animation:drive 3s infinite}
@keyframes drive{0%,100%{transform:translateX(0)}50%{transform:translateX(20px)}}
.stDataFrame{font-family:'Share Tech Mono',monospace}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  INTERFACE UTILISATEUR                                   ║
# ╚══════════════════════════════════════════════════════════╝

st.markdown("<div class='car-emoji'>🚗⚡🚗⚡🚗</div>", unsafe_allow_html=True)
st.markdown("<h1>🔋 TESLA STOCK AI PREDICTOR ⚡</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    # FIX: added options list
    model_choice = st.selectbox("🤖 Choix de l'IA", ["LSTM", "GRU", "Comparaison des deux"])
    st.markdown("---")
    st.markdown("### 🎛️ Paramètres Inférence")
    years_back = st.slider("📅 Historique global (années)", 1, 6, 6)
    lookback = st.slider(
        "🔍 Fenêtre de séquence (Lookback)", 30, 100, 60,
        help="Nombre de jours fournis à l'IA pour prédire le lendemain. "
             "Mettez la même valeur que celle utilisée pendant votre entraînement (souvent 60)."
    )
    st.markdown("---")
    st.success("✅ Vos propres modèles PyTorch sont connectés !")

with st.spinner("🔄 Téléchargement des données Tesla..."):
    df = load_tesla_data(years_back)

if df.empty:
    st.error("⚠️ Impossible de charger les données (Yahoo Finance/Rate Limit). Réessayez plus tard.")
    st.stop()

# FIX: explicitly use "Close" column everywhere
if "Close" not in df.columns:
    st.error("⚠️ Colonne 'Close' introuvable dans les données.")
    st.stop()

# Création des dates futures (jours ouvrés uniquement)
last_date = safe_ts(df.index[-1])  # FIX: correct index access
FUTURE_DATES = pd.bdate_range(start=last_date + timedelta(days=1), periods=15)

# Métriques du haut
current_price = float(df["Close"].iloc[-1])   # FIX: proper column + index
prev_price    = float(df["Close"].iloc[-2])
change        = current_price - prev_price
change_pct    = (change / prev_price) * 100
volume        = int(df["Volume"].iloc[-1])     # FIX: Volume column
high_52w      = float(df["Close"].tail(252).max())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><h3>💰 Prix Actuel</h3><h2>${current_price:.2f}</h2></div>", unsafe_allow_html=True)
with col2:
    color_sign = "🟢" if change >= 0 else "🔴"
    st.markdown(f"<div class='metric-card'><h3>📈 Variation 24h</h3><h2>{color_sign} {change:+.2f} ({change_pct:+.2f}%)</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h3>📊 Volume</h3><h2>{volume:,}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h3>🎯 Plus Haut 52 sem</h3><h2>${high_52w:.2f}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# Graphique Historique
st.markdown("<div class='tesla-card'><h2>📊 Historique — 90 derniers jours</h2>", unsafe_allow_html=True)
df_recent = df.tail(90)
fig_hist = go.Figure(go.Scatter(
    x=df_recent.index,
    y=df_recent["Close"],   # FIX: specify column
    line=dict(color="#00BFFF", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(0,191,255,0.08)"
))
fig_hist.update_layout(
    plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
    font=dict(color="white"), height=400,
    xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333")
)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  INFERENCE IA (VRAIES PRÉDICTIONS PYTORCH)               ║
# ╚══════════════════════════════════════════════════════════╝

# Préparation des données pour le réseau de neurones
prices = df["Close"].values.reshape(-1, 1)   # FIX: specify column
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# FIX: correctly extract last 'lookback' days
recent_scaled = scaled_prices.flatten()[-lookback:]

# Check we have enough data
if len(recent_scaled) < lookback:
    st.error(f"⚠️ Pas assez de données historiques ({len(recent_scaled)} jours) pour la fenêtre de {lookback} jours.")
    st.stop()

lstm_preds_real = None
gru_preds_real  = None

st.markdown("<div class='tesla-card'><h2>🧠 Génération des Prédictions (15 Jours)</h2>", unsafe_allow_html=True)

if model_choice in ("LSTM", "Comparaison des deux"):
    lstm_model = load_pytorch_model("best_tesla_LSTM_model.pt", "LSTM")
    if lstm_model:
        st.info("⚡ Inférence LSTM en cours...")
        scaled_preds = predict_future_real(lstm_model, recent_scaled, lookback=lookback, days_to_predict=15)
        lstm_preds_real = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
        st.success("✅ Prédictions LSTM générées !")
    else:
        st.warning("⚠️ Fichier 'best_tesla_LSTM_model.pt' introuvable dans le dossier courant !")

if model_choice in ("GRU", "Comparaison des deux"):
    gru_model = load_pytorch_model("best_gru_tesla_model.pt", "GRU")
    if gru_model:
        st.info("⚡ Inférence GRU en cours...")
        scaled_preds = predict_future_real(gru_model, recent_scaled, lookback=lookback, days_to_predict=15)
        gru_preds_real = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
        st.success("✅ Prédictions GRU générées !")
    else:
        st.warning("⚠️ Fichier 'best_gru_tesla_model.pt' introuvable dans le dossier courant !")

# ── Graphique comparatif des prédictions ──────────────────────────
if lstm_preds_real is not None or gru_preds_real is not None:
    df_ctx = df.tail(30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ctx.index, y=df_ctx["Close"],   # FIX: specify column
        name="Historique (30j)", line=dict(color="#00BFFF", width=3)
    ))

    # FIX: proper list concatenation — anchor last known price for continuity
    x_pred = [last_date] + list(FUTURE_DATES)

    if lstm_preds_real is not None:
        y_lstm = [current_price] + list(lstm_preds_real)   # FIX: correct concat
        fig.add_trace(go.Scatter(
            x=x_pred, y=y_lstm,
            name="AI LSTM",
            line=dict(color="#E82127", width=3, dash="dash"),
            marker=dict(size=6)
        ))

    if gru_preds_real is not None:
        y_gru = [current_price] + list(gru_preds_real)     # FIX: correct concat
        fig.add_trace(go.Scatter(
            x=x_pred, y=y_gru,
            name="AI GRU",
            line=dict(color="#00AAFF", width=3, dash="dash"),
            marker=dict(size=6)
        ))

    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line_dash="dot", line_color="yellow",
        annotation_text="Aujourd'hui", annotation_font_color="yellow"
    )
    fig.update_layout(
        plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
        font=dict(color="white"), height=500,
        hovermode="x unified",
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333", title="Prix ($)"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#E82127", borderwidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tableau des vraies prédictions ────────────────────────────
    st.markdown("### 📈 Tableau des prix projetés ($)")

    rows = []
    for i, date in enumerate(FUTURE_DATES):
        row = {"Date": date.strftime("%d %b %Y")}
        # FIX: correct dict keys with descriptive names
        if lstm_preds_real is not None:
            row["LSTM Prix"] = f"${lstm_preds_real[i]:.2f}"
            row["LSTM Δ%"]   = f"{(lstm_preds_real[i] - current_price) / current_price * 100:+.2f}%"
        if gru_preds_real is not None:
            row["GRU Prix"] = f"${gru_preds_real[i]:.2f}"
            row["GRU Δ%"]   = f"{(gru_preds_real[i] - current_price) / current_price * 100:+.2f}%"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif model_choice != "Comparaison des deux":
    st.info("💡 Aucun modèle chargé. Vérifiez que les fichiers `.pt` sont présents dans le répertoire de l'application.")

st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#666;padding:20px;font-family:Rajdhani,sans-serif;font-size:0.85em;'>
⚠️ <strong>Disclaimer</strong> : Les prédictions IA sont fournies à titre informatif uniquement.<br>
Elles ne constituent pas un conseil financier. Investir comporte des risques.
</div>
""", unsafe_allow_html=True)
