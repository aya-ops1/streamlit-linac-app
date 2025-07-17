# app_linac_pred.py – IA Maintenance LINAC + Accès sécurisé Power BI
import warnings, unicodedata, re, numpy as np
warnings.filterwarnings("ignore")
np.random.seed(42)

import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import timedelta
from pathlib import Path
from config_credentials import CREDENTIALS

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import unidecode

# ------------------ Auth ------------------
def check_login(username, password):
    return username in CREDENTIALS and CREDENTIALS[username] == password

# ------------------ Layout ------------------
st.set_page_config(page_title="Hoza | IA & Maintenance LINAC", layout="wide")
menu = st.sidebar.selectbox("📁 Menu", ["🔎 Prédiction de pannes", "📊 Rapport Power BI (sécurisé)"])

# ------------------ Helpers ------------------
def _norm(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", txt.lower())

def find_col(df: pd.DataFrame, aliases):
    wanted = [_norm(a) for a in aliases]
    for col in df.columns:
        if _norm(col) in wanted:
            return col
    raise ValueError(f"Colonne introuvable : {aliases}")

def detect_area(text):
    if pd.isna(text):
        return "Autres"
    t = unidecode.unidecode(str(text)).lower()
    if any(k in t for k in ["mlc", "multi leaf", "leaf", "collimateur"]): return "MLC"
    if any(k in t for k in ["xvi", "imagerie", "cbct", "igrt", "kv"]): return "Imagerie / XVI"
    if any(k in t for k in ["eau", "cool", "temp", "chiller", "pompe"]): return "Eau / Refroidissement"
    if any(k in t for k in ["table", "couch", "tabletop"]): return "Table patient"
    if any(k in t for k in ["magnetron", "klystron", "gun", "waveguide"]): return "Faisceau / Génération"
    if any(k in t for k in ["reseau", "informatique", "software"]): return "Informatique"
    if any(k in t for k in ["alimentation", "power", "tension"]): return "Alimentation"
    if any(k in t for k in ["mecanique", "verin", "roulement"]): return "Mécanique"
    if any(k in t for k in ["dosimet", "chambre"]): return "Dosimétrie"
    return "Autres"

# ------------------ Pipeline ------------------
def pipeline(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    col_serie = find_col(df, ["n° de serie", "numero de série"])
    col_date  = find_col(df, ["date intervention", "date d'intervention"])
    col_type  = find_col(df, ["type d'intervention"])
    col_titre = find_col(df, ["titre demande", "description"])
    col_grav  = find_col(df, ["niveau de gravité"])

    df = df[df[col_type].str.contains("corrective", case=False, na=False)].copy()
    banned = r"(maintenance|visite|check ?list|commissioning|site)"
    df = df[~df[col_type].str.lower().str.contains(banned, na=False)]
    excl_words = ["project", "training", "formation", "calibration", "installation", "test", "mise en marche"]
    df = df[~df[col_titre].str.lower().fillna("").str.contains("|".join(excl_words))]

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.dropna(subset=[col_date, col_serie]).sort_values([col_serie, col_date])
    df["Panne_15j"] = 0
    for s in df[col_serie].unique():
        m = df[col_serie] == s
        d = df.loc[m, col_date]
        for idx, d0 in d.items():
            if ((d > d0) & (d <= d0 + timedelta(days=15))).any():
                df.at[idx, "Panne_15j"] = 1

    df["Temps_depuis_dern_panne"] = -1
    for s in df[col_serie].unique():
        m = df[col_serie] == s
        d = pd.to_datetime(df.loc[m, col_date]).sort_values()
        df.loc[d.index, "Temps_depuis_dern_panne"] = (d - d.shift(1)).dt.days.fillna(-1)

    df["Gravite_dern_panne"] = None
    for s in df[col_serie].unique():
        m = df[col_serie] == s
        prev_grav = df.loc[m, col_grav].shift(1)
        df.loc[m, "Gravite_dern_panne"] = prev_grav
    df["Gravite_dern_panne"] = df["Gravite_dern_panne"].fillna("Inconnue")

    df["Freq_30jrs"] = 0
    for s in df[col_serie].unique():
        m = df[col_serie] == s
        dates = df.loc[m, col_date]
        for idx, d0 in dates.items():
            count = dates[(dates < d0) & (dates >= d0 - pd.Timedelta(days=30))].count()
            df.at[idx, "Freq_30jrs"] = count

    df["Sous_systeme"] = (df[[col_type, col_titre]].astype(str).agg(" ".join, axis=1).apply(detect_area))

    num = ["Temps_depuis_dern_panne", "Freq_30jrs"]
    cat = ["Sous_systeme", "Type d'intervention", "Niveau de gravité", "Raison sociale",
           "Famille", "Marque", "Gravite_dern_panne"]
    cat = [c for c in cat if c in df.columns]

    X = df[num + cat].copy()
    y = df["Panne_15j"]
    X[num] = X[num].fillna(-1)
    for c in cat: X[c] = X[c].fillna("Inconnu")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    tr_bal = pd.concat([X_tr, y_tr.rename("y")], axis=1)
    maj, min_ = tr_bal[tr_bal.y == 0], tr_bal[tr_bal.y == 1]
    min_over = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    bal = pd.concat([maj, min_over])
    X_tr_bal, y_tr_bal = bal.drop("y", axis=1), bal["y"]

    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)])
    rf = Pipeline([("prep", prep), ("rf", RandomForestClassifier(n_estimators=200, random_state=42))])
    rf.fit(X_tr_bal, y_tr_bal)

    report = classification_report(y_te, rf.predict(X_te), digits=2)
    imp = rf.named_steps["rf"].feature_importances_
    names = rf.named_steps["prep"].get_feature_names_out()
    imp_df = pd.DataFrame({"Feature": names, "Imp": imp}).sort_values("Imp", ascending=False).head(10)
    df["Risque_panne_15j"] = rf.predict(X)

    return df, report, imp_df, col_date, col_type

# ------------------ Panne IA ------------------
if menu == "🔎 Prédiction de pannes":
    logo_path = Path("C:/Users/Aya/Downloads/ScrimNewLogo.png")
    if logo_path.exists():
        c1, c2 = st.columns([1, 5])
        c1.image(Image.open(logo_path), width=110)
        c2.title("Générateur de prédictions IA — Maintenance LINAC")
    else:
        st.title("Générateur de prédictions IA — Maintenance LINAC")

    st.markdown(
        "<div style='font-size:17px;color:#19538a;font-weight:600;margin-top:-12px'>"
        "Bienvenue sur l'outil de prédiction du risque de panne LINAC développé par SCRIM. "
        "Chargez votre historique d’interventions ; l’IA prédit les pannes dans les 15 jours "
        "et génère un fichier enrichi.</div>",
        unsafe_allow_html=True,
    )

    file_up = st.file_uploader("Téléverser un fichier Excel", type="xlsx")
    if file_up:
        st.info("⌛ Traitement en cours…")
        try:
            df_pred, rep_txt, imp_df, col_date, col_type = pipeline(pd.read_excel(file_up))
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.stop()

        st.success("✅ Algorithme retenu : Random Forest (200 arbres, seed 42)")

        total = len(df_pred)
        risk_n = int((df_pred["Risque_panne_15j"] == 1).sum())
        risk_pct = risk_n / total * 100
        k1, k2, k3 = st.columns(3)
        k1.metric("Machines à risque", risk_n)
        k2.metric("Machines analysées", total)
        k3.metric("% à risque", f"{risk_pct:.1f}%")

        if "Raison sociale" in df_pred.columns and risk_n:
            top = df_pred[df_pred.Risque_panne_15j == 1]["Raison sociale"].value_counts().head(3)
            rows = []
            for i, (site, n) in enumerate(zip(top.index, top.values), 1):
                row = df_pred[(df_pred["Raison sociale"] == site) & (df_pred.Risque_panne_15j == 1)].sort_values(col_date).iloc[-1]
                d_prob = (pd.to_datetime(row[col_date]) + timedelta(days=15)).date()
                rows.append({
                    "Rang": i,
                    "Raison sociale": site,
                    "Sous-système": row["Sous_systeme"],
                    "Type pressenti": row[col_type],
                    "Date probable": d_prob
                })
            st.info("### Top 3 établissements à risque (15 j) :")
            st.table(pd.DataFrame(rows))
        else:
            st.info("Aucun établissement n’est actuellement classé à risque.")

        st.markdown("---")
        st.subheader("Top 10 variables influentes")
        fig, ax = plt.subplots(figsize=(6, 4))
        imp_df.iloc[::-1].plot.barh(x="Feature", y="Imp", ax=ax, color="#1f77b4", legend=False)
        ax.set_xlabel("Importance (Gini)")
        ax.set_ylabel("")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Tableau complet (aperçu)")
        st.dataframe(df_pred, use_container_width=True)

        buff = BytesIO()
        df_pred.to_excel(buff, index=False)
        buff.seek(0)
        st.download_button("💾 Télécharger le fichier enrichi", buff, "Predictions_LINAC.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------ Rapport Power BI sécurisé ------------------
elif menu == "📊 Rapport Power BI (sécurisé)":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🔐 Connexion requise")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.error("Identifiants incorrects.")
    else:
        st.success("Connexion réussie ✅")
        st.markdown("### Rapport Power BI — Suivi de la maintenance des LINAC")
        st.components.v1.html(
            '''
            <iframe width="100%" height="800px"
            src="https://app.powerbi.com/groups/me/reports/8ba6eea6-9a68-464c-89d7-98da2c432144/7b6a902fa29bc3822313?experience=power-bi"
            frameborder="0" allowFullScreen="true"></iframe>
            ''',
            height=850,
            scrolling=True
        )
