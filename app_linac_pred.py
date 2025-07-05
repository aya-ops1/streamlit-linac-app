# app_linac_pred.py  –  IA Maintenance LINAC (RF + oversampling + sous-système)
# ---------------------------------------------------------------------------
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

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# 1. Helpers : normaliser texte & trouver colonne
# ---------------------------------------------------------------------------
def _norm(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", txt.lower())

def find_col(df: pd.DataFrame, aliases):
    wanted = [_norm(a) for a in aliases]
    for col in df.columns:
        if _norm(col) in wanted:
            return col
    raise ValueError(f"Colonne introuvable : {aliases}")

# ---------------------------------------------------------------------------
# 2. Détection du sous-système à partir d’un texte
# ---------------------------------------------------------------------------
import unidecode

def detect_area(text):
    """
    Détection du sous-système (catégorie de panne) à partir d'un texte (titre de la demande, type, description...).
    Retourne un libellé homogène utilisable pour les statistiques/affichage.
    """
    if pd.isna(text):
        return "Autres"
    t = unidecode.unidecode(str(text)).lower()
    # MLC
    if any(k in t for k in ["mlc", "multi leaf", "multileaf", "leaf", "lame", "collimateur", "leaf missing", "carte mlc"]):
        return "MLC"
    # Imagerie
    if any(k in t for k in ["xvi", "imagerie", "image", "portal", "cbct", "igrt", "kvcb", "kv", "mvi", "imaging"]):
        return "Imagerie / XVI"
    # Eau / Cooling
    if any(k in t for k in ["eau", "water", "cool", "temp", "chiller", "groupe froid", "pompe", "water temp"]):
        return "Eau / Refroidissement"
    # Table
    if any(k in t for k in ["table", "couch", "tabletop", "vertical", "mouvement table", "support table"]):
        return "Table patient"
    # Faisceau
    if any(k in t for k in ["magnetron", "klystron", "gun", "head", "tete", "faisceau", "waveguide", "modulateur", "rf"]):
        return "Faisceau / Génération"
    # Informatique / réseau
    if any(k in t for k in ["reseau", "network", "informatique", "software", "logiciel", "licence", "license", "ordi", "computer"]):
        return "Informatique"
    # Alimentation / Electricité
    if any(k in t for k in ["alimentation", "power", "panne electrique", "electrical", "fuse", "fusible", "tension"]):
        return "Alimentation"
    # Mécanique générale
    if any(k in t for k in ["mecanique", "verin", "axe", "roulement", "motor", "moteur", "transmission"]):
        return "Mécanique"
    # Dosimétrie
    if any(k in t for k in ["dosimet", "dosimetry", "electrometer", "chambre"]):
        return "Dosimétrie"
    # Autres
    return "Autres"
# ---------------------------------------------------------------------------
# 3. Mise en page Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predictions Pannes LINAC", layout="wide")

logo_path = Path("C:/Users/Aya/Downloads/ScrimNewLogo.png")
if logo_path.exists():
    c1, c2 = st.columns([1, 5])
    c1.image(Image.open(logo_path), width=110)
    c2.title("Générateur de prédictions IA — Maintenance LINAC")
else:
    st.title("Générateur de prédictions IA — Maintenance LINAC")

st.markdown(
    "<div style='font-size:17px;color:#19538a;font-weight:600;margin-top:-12px'>"
    "Chargez votre historique d’interventions ; l’IA prédit les pannes dans les 15 jours "
    "et génère un fichier enrichi.</div>",
    unsafe_allow_html=True,
)

file_up = st.file_uploader("Téléverser un fichier Excel", type="xlsx")

# ---------------------------------------------------------------------------
# 4. Pipeline principal
# ---------------------------------------------------------------------------
def pipeline(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    col_serie = find_col(df, ["n° de serie","N° de série", "N° de serie", "numero de série"])
    col_date  = find_col(df, ["date intervention", "date d'intervention", "Date Intervention"])
    col_type  = find_col(df, ["type d'intervention","type d'intervention'", "type intervention"])
    col_titre = find_col(df, ["titre demande", "description","titre de la demande", "titre de demande"])
    col_grav  = find_col(df, ["niveau de gravité","Niveau de gravité"])

    # ----- étape 1 : filtrer Corrective
    df = df[df[col_type].str.contains("corrective", case=False, na=False)].copy()

    # ----- étape 2 : exclure maintenance / visite / check-list / commissioning / site
    banned = r"(maintenance|visite|check ?list|commissioning|site)"
    mask_bad = df[col_type].str.lower().str.contains(banned, na=False)
    mask_keep = df[col_type].str.lower().str.contains("corrective", na=False)
    df = df[~mask_bad | mask_keep]

    # ----- étape 3 : exclusions supplémentaires sur le titre
    excl_words = ["project", "training", "formation", "commissioning", "calibration", "installation",
            "test", "mise en marche", "verification", "checklist", "visite"]
    df = df[~df[col_titre].str.lower().fillna("").str.contains("|".join(excl_words))]

    # ----- étape 4 : cible
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.dropna(subset=[col_date, col_serie]).sort_values([col_serie, col_date])
    df["Panne_15j"] = 0
    for s in df[col_serie].unique():
        m = df[col_serie] == s
        d = df.loc[m, col_date]
        for idx, d0 in d.items():
            if ((d > d0) & (d <= d0 + timedelta(days=15))).any():
                df.at[idx, "Panne_15j"] = 1

    # ----- étape 5 : feature temps depuis panne précédente
    df["Temps_depuis_dern_panne"] = -1
    for s in df[col_serie].unique():
        m = df[col_serie] == s
        d = pd.to_datetime(df.loc[m, col_date]).sort_values()
        df.loc[d.index, "Temps_depuis_dern_panne"] = (d - d.shift(1)).dt.days.fillna(-1)

 # --- Feature 2 : Gravité de la panne précédente
    df["Gravite_dern_panne"] = None
    for s in df[col_serie].unique():
        m = df[col_serie] == s
        prev_grav = df.loc[m, col_grav].shift(1)
        df.loc[m, "Gravite_dern_panne"] = prev_grav
    df["Gravite_dern_panne"] = df["Gravite_dern_panne"].fillna("Inconnue")

    # --- Feature 3 : Fréquence incidents sur 30j
    df["Freq_30jrs"] = 0
    for s in df[col_serie].unique():
        m = df[col_serie] == s
        dates = df.loc[m, col_date]
        for idx, d0 in dates.items():
            count = dates[(dates < d0) & (dates >= d0 - pd.Timedelta(days=30))].count()
            df.at[idx, "Freq_30jrs"] = count

    # ----- étape 6 : détection sous-système
    df["Sous_systeme"] = (df[[col_type, col_titre]]
                          .astype(str)
                          .agg(" ".join, axis=1)
                          .apply(detect_area))

    # ----- préparation scikit-learn
    num = ["Temps_depuis_dern_panne", "Freq_30jrs"]
    cat = ["Sous_systeme",
           "Type d'intervention", "Niveau de gravité",
           "Raison sociale", "Famille", "Marque", "Gravite_dern_panne"]
    cat = [c for c in cat if c in df.columns]

    X = df[num + cat].copy()
    y = df["Panne_15j"]
    X[num] = X[num].fillna(-1)
    for c in cat: X[c] = X[c].fillna("Inconnu")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.30, random_state=42
    )

    # ----- sur-échantillonnage
    tr_bal = pd.concat([X_tr, y_tr.rename("y")], axis=1)
    maj, min_ = tr_bal[tr_bal.y == 0], tr_bal[tr_bal.y == 1]
    min_over = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    bal = pd.concat([maj, min_over])
    X_tr_bal, y_tr_bal = bal.drop("y", axis=1), bal["y"]

    # ----- Random Forest
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", "passthrough", num)])
    rf = Pipeline([("prep", prep),
                   ("rf", RandomForestClassifier(n_estimators=200, random_state=42))])
    rf.fit(X_tr_bal, y_tr_bal)

    report = classification_report(y_te, rf.predict(X_te), digits=2)

    # importance
    imp = rf.named_steps["rf"].feature_importances_
    names = rf.named_steps["prep"].get_feature_names_out()
    imp_df = (pd.DataFrame({"Feature": names, "Imp": imp})
                .sort_values("Imp", ascending=False).head(10))

    # prédiction full set
    df["Risque_panne_15j"] = rf.predict(X)

    return df, report, imp_df, col_date, col_type


# ---------------------------------------------------------------------------
# 5. Interface Streamlit
# ---------------------------------------------------------------------------
if file_up:
    st.info("⌛ Traitement en cours…")
    try:
        df_pred, rep_txt, imp_df, col_date, col_type = pipeline(
            pd.read_excel(file_up)
        )
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()

    st.success("✅ Algorithme retenu : Random Forest (200 arbres, seed 42)")

    # KPI
    total = len(df_pred)
    risk_n = int((df_pred["Risque_panne_15j"] == 1).sum())
    risk_pct = risk_n / total * 100
    k1, k2, k3 = st.columns(3)
    k1.metric("Machines à risque", risk_n)
    k2.metric("Machines analysées", total)
    k3.metric("% à risque", f"{risk_pct:.1f}%")

    # Top-3 établissements
    if "Raison sociale" in df_pred.columns and risk_n:
        top = (df_pred[df_pred.Risque_panne_15j == 1]["Raison sociale"]
               .value_counts().head(3))
        rows = []
        for i, (site, n) in enumerate(zip(top.index, top.values), 1):
            row = (df_pred[(df_pred["Raison sociale"] == site) &
                           (df_pred.Risque_panne_15j == 1)]
                   .sort_values(col_date).iloc[-1])
            d_prob = (pd.to_datetime(row[col_date]) + timedelta(days=15)).date()
            rows.append({
                "Rang": i,
                "Raison sociale": site,
                "Sous-système":   row["Sous_systeme"],
                "Type pressenti": row[col_type],
                "Date probable":  d_prob
            })
        st.info("### Top 3 établissements à risque (15 j) :")
        st.table(pd.DataFrame(rows))
    else:
        st.info("Aucun établissement n’est actuellement classé à risque.")

    st.markdown("---")
    st.subheader("Top 10 variables influentes")
    fig, ax = plt.subplots(figsize=(6,4))
    imp_df.iloc[::-1].plot.barh(x="Feature", y="Imp",
                                ax=ax, color="#1f77b4", legend=False)
    ax.set_xlabel("Importance (Gini)")
    ax.set_ylabel("")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Tableau complet (aperçu)")
    st.dataframe(df_pred, use_container_width=True)

    buff = BytesIO()
    df_pred.to_excel(buff, index=False)
    buff.seek(0)
    st.download_button(
        label="💾 Télécharger le fichier enrichi",
        data=buff,
        file_name="Extract_Ayoub_with_pred.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
