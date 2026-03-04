"""
Application Streamlit - Lead Scoring ML
Auteur: Bi Irié Albéric TRA
"""
import streamlit as st
import pandas as pd
from data.generate_leads import generate_lead_data
from models.lead_scorer import LeadScorer
from utils.metrics import (
    plot_roc_curve, plot_feature_importance,
    plot_score_distribution, plot_confusion_matrix
)

st.set_page_config(page_title="Lead Scoring ML", layout="wide")

# Titre
st.title("🎯 Lead Scoring ML - CRM Analytics")
st.markdown("**Projet Data Science** | École Centrale Casablanca | Bi Irié Albéric TRA")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
n_leads = st.sidebar.slider("Nombre de leads", 1000, 10000, 5000, 500)
model_type = st.sidebar.selectbox("Modèle ML", ['random_forest', 'xgboost'])

# Génération données
with st.spinner("Génération des données..."):
    df = generate_lead_data(n_leads=n_leads)

# Section 1: Données
st.header("📊 Données Leads")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Leads", f"{len(df):,}")
with col2:
    st.metric("Taux Conversion", f"{df['converted'].mean():.1%}")
with col3:
    st.metric("Temps Moyen Site", f"{df['time_on_site'].mean():.1f} min")

with st.expander("📋 Aperçu des données"):
    st.dataframe(df.head(20))

# Section 2: Entraînement Modèle
st.header("🤖 Entraînement Modèle ML")

model = LeadScorer(model_type=model_type)

with st.spinner(f"Entraînement {model_type}..."):
    metrics, X_test, y_test, y_pred_proba = model.train(df)

# Métriques
st.subheader("📈 Performance Modèle")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
with col2:
    st.metric("Precision", f"{metrics['precision']:.3f}")
with col3:
    st.metric("Recall", f"{metrics['recall']:.3f}")
with col4:
    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

st.info(f"✅ Cross-validation AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")

# Visualisations métriques
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(plot_roc_curve(y_test, y_pred_proba), use_container_width=True)

with col2:
    y_pred = (y_pred_proba > 0.5).astype(int)
    st.plotly_chart(plot_confusion_matrix(y_test, y_pred), use_container_width=True)

# Feature importance
st.subheader("🔍 Importance des Features")
st.plotly_chart(plot_feature_importance(model.feature_importance), use_container_width=True)

# Section 3: Scoring Leads
st.header("💯 Scoring des Leads")

with st.spinner("Calcul scores..."):
    scores_df = model.score_leads(df)

# Distribution scores
st.plotly_chart(plot_score_distribution(scores_df), use_container_width=True)

# Statistiques par catégorie
st.subheader("📊 Statistiques par Catégorie")
category_stats = scores_df['categorie'].value_counts()

col1, col2, col3 = st.columns(3)
with col1:
    chaud = category_stats.get('Chaud', 0)
    st.metric("🔥 Leads Chauds (≥70)", f"{chaud:,}", f"{chaud/len(scores_df):.1%}")
with col2:
    tiede = category_stats.get('Tiède', 0)
    st.metric("🌡️ Leads Tièdes (40-69)", f"{tiede:,}", f"{tiede/len(scores_df):.1%}")
with col3:
    froid = category_stats.get('Froid', 0)
    st.metric("❄️ Leads Froids (<40)", f"{froid:,}", f"{froid/len(scores_df):.1%}")

# Top leads
st.subheader("🏆 Top 20 Leads à Contacter en Priorité")
top_leads = scores_df.sort_values('score', ascending=False).head(20)

# Merge avec données originales
top_leads_full = top_leads.merge(df, on='lead_id')
st.dataframe(top_leads_full[['lead_id', 'score', 'categorie', 'time_on_site', 
                              'email_clicks', 'page_views', 'source']], use_container_width=True)

# Section 4: Impact Business
st.header("💼 Impact Business & ROI")

col1, col2, col3 = st.columns(3)

with col1:
    chauds = len(scores_df[scores_df['categorie'] == 'Chaud'])
    st.metric("Leads Chauds à Prioriser", f"{chauds:,}")

with col2:
    conversion_rate_chauds = 0.45
    conversions_estimated = int(chauds * conversion_rate_chauds)
    st.metric("Conversions Estimées (45%)", f"{conversions_estimated:,}")

with col3:
    avg_deal_value = 5000  # MAD
    revenue_potential = conversions_estimated * avg_deal_value
    st.metric("Revenu Potentiel", f"{revenue_potential:,} MAD")

st.caption("*Hypothèses : 45% conversion leads chauds, 5,000 MAD par deal moyen")

# Footer
st.markdown("---")
st.markdown("**Projet réalisé par Bi Irié Albéric TRA** | École Centrale Casablanca | 2025")
st.markdown(f"Stack : Python, {model_type}, scikit-learn, Streamlit, Plotly")
