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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- NOUVEAUX IMPORTS POUR L'IA RÉELLE ---
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.gru(x, (h0.detach()))
        out = self.fc(out)
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
        
        # Si le fichier contient juste les poids (state_dict)
        if isinstance(model_data, dict) or isinstance(model_data, collections.OrderedDict):
            if model_type == "LSTM":
                model = LSTMModel()
            else:
                model = GRUModel()
            model.load_state_dict(model_data)
        else:
            # Si le fichier contient le modèle entier
            model = model_data
            
        model.eval() # Mode évaluation
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {model_type}: {e}")
        return None

def predict_future_real(model, recent_data_scaled, days_to_predict=15):
    """Effectue des prédictions jour par jour (fenêtre glissante autorégressive)."""
    predictions =[]
    current_seq = recent_data_scaled.copy()
    
    with torch.no_grad():
        for _ in range(days_to_predict):
            # Transformation en Tenseur PyTorch: shape (1, lookback, 1)
            x_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # Prédiction du jour suivant
            pred = model(x_tensor).item()
            predictions.append(pred)
            
            # Décalage de la fenêtre : on retire le 1er jour, on ajoute la nouvelle prédiction à la fin
            current_seq = np.append(current_seq, pred)
            
    return np.array(predictions)


# ╔══════════════════════════════════════════════════════════╗
# ║  DONNÉES HISTORIQUES (AVEC ANTI RATE-LIMIT YAHOO)        ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(show_spinner=False, ttl=3600)
def load_tesla_data(years_back: int = 6) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years_back * 365)
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    
    retries = Retry(total=5, backoff_factor=1, status_forcelist=)
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        tsla = yf.Ticker("TSLA", session=session)
        df = tsla.history(start=start, end=end)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
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
.main,.stApp{background:linear-gradient(135deg,#0a0a0a 0%,#1a1a1a 100%)}
h1{color:#E82127;font-family:'Gotham',sans-serif;text-align:center;font-size:3.5em; font-weight:bold;text-shadow:2px 2px 4px rgba(232,33,39,.3);padding:20px}
h2,h3{color:#fff;font-family:'Gotham',sans-serif}
.tesla-card{background:linear-gradient(135deg,#1a1a1a 0%,#2a2a2a 100%);border-radius:15px;padding:25px;margin:15px 0;border:2px solid #E82127;box-shadow:0 8px 16px rgba(232,33,39,.2)}
.metric-card{background:linear-gradient(135deg,#E82127 0%,#C41E23 100%);border-radius:10px;padding:20px;text-align:center;color:#fff;margin:10px;box-shadow:0 4px 8px rgba(0,0,0,.3)}
.car-emoji{font-size:3em;text-align:center;animation:drive 3s infinite}
@keyframes drive{0%,100%{transform:translateX(0)}50%{transform:translateX(20px)}}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  INTERFACE UTILISATEUR                                   ║
# ╚══════════════════════════════════════════════════════════╝

st.markdown("<div class='car-emoji'>🚗⚡🚗⚡🚗</div>", unsafe_allow_html=True)
st.markdown("<h1>🔋 TESLA STOCK AI PREDICTOR ⚡</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    model_choice = st.selectbox("🤖 Choix de l'IA",)
    st.markdown("---")
    st.markdown("### 🎛️ Paramètres Inférence")
    years_back = st.slider("📅 Historique global (années)", 1, 6, 6)
    lookback = st.slider("🔍 Fenêtre de séquence (Lookback)", 30, 100, 60, help="Nombre de jours fournis à l'IA pour prédire le lendemain. Mettez la même valeur que celle utilisée pendant votre entraînement (souvent 60).")
    st.markdown("---")
    st.success("✅ Vos propres modèles PyTorch sont connectés !")

with st.spinner("🔄 Téléchargement des données Tesla..."):
    df = load_tesla_data(years_back)

if df.empty:
    st.error("⚠️ Impossible de charger les données (Yahoo Finance/Rate Limit). Réessayez plus tard.")
    st.stop()

# Création des dates futures (uniquement les jours ouvrés)
last_date = safe_ts(df.index)
FUTURE_DATES = pd.bdate_range(start=last_date + timedelta(days=1), periods=15)

# Métriques du haut
current_price = float(df.iloc)
prev_price    = float(df.iloc)
change        = current_price - prev_price
change_pct    = (change / prev_price) * 100
volume        = int(df.iloc)
high_52w      = float(df.tail(252).max())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><h3>💰 Prix Actuel</h3><h2>${current_price:.2f}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h3>📈 Variation 24h</h3><h2>{change:+.2f} ({change_pct:+.2f}%)</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h3>📊 Volume</h3><h2>{volume:,}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h3>🎯 Plus Haut 52 sem</h3><h2>${high_52w:.2f}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# Graphique Historique
st.markdown("<div class='tesla-card'><h2>📊 Historique — 90 derniers jours</h2>", unsafe_allow_html=True)
df_recent = df.tail(90)
fig_hist = go.Figure(go.Scatter(x=df_recent.index, y=df_recent, line=dict(color="#00BFFF", width=2.5), fill="tozeroy"))
fig_hist.update_layout(plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a", font=dict(color="white"), height=400)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  INFERENCE IA (VRAIES PRÉDICTIONS PYTORCH)               ║
# ╚══════════════════════════════════════════════════════════╝

# Préparation des données pour le réseau de neurones
prices = df.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# On extrait les X derniers jours correspondants au 'lookback' (par défaut 60)
recent_scaled = scaled_prices.flatten()

lstm_preds_real = None
gru_preds_real  = None

st.markdown("<div class='tesla-card'><h2>🧠 Génération des Prédictions (15 Jours)</h2>", unsafe_allow_html=True)

if model_choice in ("LSTM", "Comparaison des deux"):
    lstm_model = load_pytorch_model("best_tesla_LSTM_model.pt", "LSTM")
    if lstm_model:
        st.info("⚡ Inférence LSTM en cours...")
        scaled_preds = predict_future_real(lstm_model, recent_scaled, days_to_predict=15)
        # On dénormalise pour retrouver les dollars ($)
        lstm_preds_real = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
    else:
        st.warning("⚠️ Fichier 'best_tesla_LSTM_model.pt' introuvable dans le dossier GitHub !")

if model_choice in ("GRU", "Comparaison des deux"):
    gru_model = load_pytorch_model("best_gru_tesla_model.pt", "GRU")
    if gru_model:
        st.info("⚡ Inférence GRU en cours...")
        scaled_preds = predict_future_real(gru_model, recent_scaled, days_to_predict=15)
        gru_preds_real = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
    else:
        st.warning("⚠️ Fichier 'best_gru_tesla_model.pt' introuvable dans le dossier GitHub !")

# ── Graphique comparatif des prédictions réelles ──────────────────
if lstm_preds_real is not None or gru_preds_real is not None:
    df_ctx = df.tail(30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ctx.index, y=df_ctx, name="Historique (30j)", line=dict(color="#00BFFF", width=3)))

    # Coordonnées X (on relie le dernier jour connu aux 15 jours suivants)
    x_pred = + list(FUTURE_DATES)

    if lstm_preds_real is not None:
        y_lstm = + list(lstm_preds_real)
        fig.add_trace(go.Scatter(x=x_pred, y=y_lstm, name="AI LSTM", line=dict(color="#E82127", width=3, dash="dash"), marker=dict(size=6)))

    if gru_preds_real is not None:
        y_gru = + list(gru_preds_real)
        fig.add_trace(go.Scatter(x=x_pred, y=y_gru, name="AI GRU", line=dict(color="#00AAFF", width=3, dash="dash"), marker=dict(size=6)))

    fig.add_vline(x=last_date.timestamp() * 1000, line_dash="dot", line_color="yellow", annotation_text="Aujourd'hui")
    fig.update_layout(plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a", font=dict(color="white"), height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ── Tableaux des vraies prédictions ──────────────────
    st.markdown("### 📈 Tableau des prix projetés ($)")
    
    rows =[]
    for i, date in enumerate(FUTURE_DATES):
        row = {"Date": date.strftime("%d %b %Y")}
        if lstm_preds_real is not None:
            row = f"${lstm_preds_real:.2f}"
            row = f"{(lstm_preds_real - current_price)/current_price*100:+.2f}%"
        if gru_preds_real is not None:
            row = f"${gru_preds_real:.2f}"
            row = f"{(gru_preds_real - current_price)/current_price*100:+.2f}%"
        rows.append(row)
        
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)
