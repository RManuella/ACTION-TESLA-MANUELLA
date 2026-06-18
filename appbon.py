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


# Image voiture Tesla encodée en base64
TESLA_LOGO_B64 = "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"


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

def infer_dims_from_state_dict(state_dict, model_type):
    """Déduit automatiquement hidden_dim et num_layers depuis les poids sauvegardés."""
    key = "lstm.weight_hh_l0" if model_type == "LSTM" else "gru.weight_hh_l0"
    if key in state_dict:
        # weight_hh_l0 shape: (num_gates * hidden_dim, hidden_dim)
        # hidden_dim = nb de colonnes (dimension cachée réelle)
        hidden_dim = state_dict[key].shape[1]
        # Compter le nombre de couches
        num_layers = 0
        layer_prefix = "lstm" if model_type == "LSTM" else "gru"
        while f"{layer_prefix}.weight_hh_l{num_layers}" in state_dict:
            num_layers += 1
        return hidden_dim, max(num_layers, 1)
    return 50, 2  # valeurs par défaut de secours


@st.cache_resource
def load_pytorch_model(model_path, model_type):
    """Charge le modèle PyTorch en détectant automatiquement l'architecture."""
    if not os.path.exists(model_path):
        return None

    try:
        model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

        if isinstance(model_data, (dict, collections.OrderedDict)):
            # Auto-détection des dimensions depuis le checkpoint
            hidden_dim, num_layers = infer_dims_from_state_dict(model_data, model_type)
            st.info(f"🔍 Architecture détectée — {model_type}: hidden_dim={hidden_dim}, num_layers={num_layers}")

            if model_type == "LSTM":
                model = LSTMModel(hidden_dim=hidden_dim, num_layers=num_layers)
            else:
                model = GRUModel(hidden_dim=hidden_dim, num_layers=num_layers)
            model.load_state_dict(model_data)
        else:
            # Modèle complet sauvegardé directement
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
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

.main,.stApp{background:#080808}
h2,h3{color:#fff;font-family:'Rajdhani',sans-serif}
.tesla-card{background:linear-gradient(135deg,#1a1a1a 0%,#2a2a2a 100%);border-radius:15px;
   padding:25px;margin:15px 0;border:2px solid #E82127;box-shadow:0 8px 16px rgba(232,33,39,.2)}
.metric-card{background:linear-gradient(135deg,#E82127 0%,#C41E23 100%);border-radius:10px;
   padding:20px;text-align:center;color:#fff;margin:10px;box-shadow:0 4px 8px rgba(0,0,0,.3)}
.metric-card h3{font-family:'Rajdhani',sans-serif;font-size:0.9em;opacity:0.85;margin:0 0 8px 0}
.metric-card h2{font-family:'Share Tech Mono',monospace;font-size:1.4em;margin:0}
.stDataFrame{font-family:'Share Tech Mono',monospace}

/* ── HERO HEADER ── */
.hero-wrapper {
    position: relative;
    width: 100%;
    padding: 48px 0 36px 0;
    text-align: center;
    overflow: hidden;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(232,33,39,0.12) 0%, transparent 70%),
                linear-gradient(180deg, #0d0d0d 0%, #080808 100%);
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 2rem;
}
/* grille de fond subtile */
.hero-wrapper::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(232,33,39,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(232,33,39,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}
.hero-logo {
    width: 120px;
    margin-bottom: 24px;
    animation: logoPulse 4s ease-in-out infinite;
    filter: drop-shadow(0 0 22px rgba(232,33,39,0.75));
    position: relative; z-index: 1;
}
@keyframes logoPulse {
    0%,100% { transform: scale(1);   filter: drop-shadow(0 0 18px rgba(232,33,39,0.65)); }
    50%      { transform: scale(1.05); filter: drop-shadow(0 0 34px rgba(232,33,39,0.95)); }
}
.hero-eyebrow {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72em;
    letter-spacing: 0.35em;
    color: #E82127;
    text-transform: uppercase;
    margin-bottom: 12px;
    position: relative; z-index: 1;
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.8em, 4.5vw, 3.4em);
    font-weight: 900;
    color: #fff;
    letter-spacing: 0.04em;
    line-height: 1.1;
    margin: 0 auto 10px auto;
    position: relative; z-index: 1;
}
.hero-title .accent { color: #E82127; }
.hero-divider {
    width: 80px; height: 2px;
    background: linear-gradient(90deg, transparent, #E82127, transparent);
    margin: 14px auto 16px auto;
    position: relative; z-index: 1;
    animation: dividerGlow 2.5s ease-in-out infinite;
}
@keyframes dividerGlow {
    0%,100% { width: 80px; opacity: 0.6; }
    50%      { width: 140px; opacity: 1; }
}
.hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.0em;
    color: #666;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    position: relative; z-index: 1;
}
.hero-badges {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 20px;
    flex-wrap: wrap;
    position: relative; z-index: 1;
}
.hero-author {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95em;
    color: #555;
    letter-spacing: 0.12em;
    margin-top: 14px;
    margin-bottom: 0;
    position: relative; z-index: 1;
}
.hero-author span {
    color: #E82127;
    font-weight: 700;
}
.hero-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68em;
    color: #E82127;
    border: 1px solid rgba(232,33,39,0.35);
    border-radius: 2px;
    padding: 4px 14px;
    letter-spacing: 0.15em;
    background: rgba(232,33,39,0.06);
}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  INTERFACE UTILISATEUR                                   ║
# ╚══════════════════════════════════════════════════════════╝

st.markdown(f"""
<div class="hero-wrapper">
    <img class="hero-logo" src="data:image/png;base64,{TESLA_LOGO_B64}" />
    <div class="hero-eyebrow">NYSE : TSLA &nbsp;·&nbsp; Analyse par Intelligence Artificielle</div>
    <div class="hero-title">PRÉDICTEUR <span class="accent">IA</span><br>DU COURS TESLA</div>
    <div class="hero-author">By <span>EFEMBA Manuella</span></div>
    <div class="hero-divider"></div>
    <div class="hero-subtitle">Prévision 15 jours · Deep Learning · LSTM &amp; GRU</div>
    <div class="hero-badges">
        <span class="hero-badge">LSTM</span>
        <span class="hero-badge">GRU</span>
        <span class="hero-badge">PyTorch</span>
        <span class="hero-badge">Temps réel</span>
    </div>
</div>
""", unsafe_allow_html=True)

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
    if not os.path.exists("best_tesla_LSTM_model.pt"):
        st.warning("⚠️ Fichier 'best_tesla_LSTM_model.pt' introuvable dans le dossier courant !")
    else:
        lstm_model = load_pytorch_model("best_tesla_LSTM_model.pt", "LSTM")
        if lstm_model:
            st.info("⚡ Inférence LSTM en cours...")
            scaled_preds = predict_future_real(lstm_model, recent_scaled, lookback=lookback, days_to_predict=15)
            lstm_preds_real = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
            st.success("✅ Prédictions LSTM générées !")

if model_choice in ("GRU", "Comparaison des deux"):
    if not os.path.exists("best_gru_tesla_model.pt"):
        st.warning("⚠️ Fichier 'best_gru_tesla_model.pt' introuvable dans le dossier courant !")
    else:
        gru_model = load_pytorch_model("best_gru_tesla_model.pt", "GRU")
        if gru_model:
            st.info("⚡ Inférence GRU en cours...")
            scaled_preds = predict_future_real(gru_model, recent_scaled, lookback=lookback, days_to_predict=15)
            gru_preds_real = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
            st.success("✅ Prédictions GRU générées !")

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