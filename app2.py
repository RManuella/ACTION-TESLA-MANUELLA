"""
Tesla Stock Predictor — Streamlit App
Prédictions 15 jours issues de vos modèles PyTorch (LSTM et GRU)
"""

import time
import warnings
import os
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
# Note: Si vos modèles ont été entraînés avec une architecture différente, 
# vous pouvez ajuster 'hidden_dim' ou 'num_layers' ici.

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
    """Charge le modèle PyTorch. Gère à la fois le modèle complet et le state_dict."""
    if not os.path.exists(model_path):
        return None
        
    try:
        # Essaye de charger le modèle complet
        model = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(model, dict):
            # Si c'est un dictionnaire, c'est un state_dict
            if model_type == "LSTM":
                model = LSTMModel()
            else:
                model = GRUModel()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        model.eval() # Mode évaluation
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {model_type}: {e}")
        return None

def predict_future_real(model, recent_data_scaled, days_to_predict=15):
    """Effectue des prédictions jour par jour (autorégressif)."""
    predictions =[]
    # Création d'une copie pour ne pas modifier la donnée d'origine
    current_seq = recent_data_scaled.copy()
    
    with torch.no_grad():
        for _ in range(days_to_predict):
            # Format attendu par PyTorch: (batch_size, seq_len, features) -> (1, lookback, 1)
            x_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # Prédiction du prochain jour
            pred = model(x_tensor).item()
            predictions.append(pred)
            
            # Mise à jour de la séquence: on enlève le 1er jour, on ajoute la prédiction
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
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
    lookback = st.slider("🔍 Fenêtre de séquence (Lookback)", 30, 100, 60, help="Nombre de jours fournis à l'IA pour prédire le lendemain. (Généralement 60 jours en LSTM)")
    st.markdown("---")
    st.success("✅ Architecture PyTorch intégrée")

with st.spinner("🔄 Téléchargement des données Tesla..."):
    df = load_tesla_data(years_back)

if df.empty:
    st.error("⚠️ Impossible de charger les données (Yahoo Finance).")
    st.stop()

# Dynamiser les futures dates à partir d'aujourd'hui
last_date = df.index
FUTURE_DATES = pd.bdate_range(start=last_date + timedelta(days=1), periods=15)

# Métriques rapides
current_price = float(df.iloc)
prev_price    = float(df.iloc)
change        = current_price - prev_price
change_pct    = (change / prev_price) * 100
volume        = int(df.iloc)
high_52w      = float(df.tail(252).max())

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='metric-card'><h3>💰 Prix Actuel</h3><h2>${current_price:.2f}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>📈 Variation 24h</h3><h2>{change:+.2f} ({change_pct:+.2f}%)</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>📊 Volume</h3><h2>{volume:,}</h2></div
