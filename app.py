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

st.set_page_config(
    page_title="Lead Scoring ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé - Design Premium
st.markdown("""
<style>
    /* Main content styling */
    .main {
        padding: 2rem 3rem;
    }
    
    /* Header styling */
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e8eaed;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #e8eaed;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #30363d;
        padding-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.3rem;
        font-weight: 500;
        color: #c9d1d9;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1f26;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar {
        background-color: #0f1419;
        padding: 2rem 1.5rem;
    }
    
    .sidebar-content {
        color: #e8eaed;
    }
    
    /* Text styling */
    p, span {
        color: #c9d1d9;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1f77b4;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2a8bd9;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    
    /* Info boxes */
    [data-testid="stInfo"] {
        background-color: #1a1f26;
        border-left: 4px solid #1f77b4;
        border-radius: 6px;
        color: #c9d1d9;
    }
    
    /* Error boxes */
    [data-testid="stError"] {
        background-color: #1a1f26;
        border-left: 4px solid #f85149;
        border-radius: 6px;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# Titre
st.markdown("<h1>Lead Scoring ML</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.95rem; color: #8b949e; margin-top: -1rem;'>CRM Analytics & Predictive Lead Scoring | École Centrale Casablanca</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.markdown("<h3 style='margin-top: 0;'>Configuration</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    n_leads = st.slider("Nombre de leads", 1000, 10000, 5000, 500)
    model_type = st.selectbox("Modèle ML", ['random_forest', 'xgboost'])
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.85rem; color: #8b949e;'>Premium ML Analytics | v1.0</p>", unsafe_allow_html=True)

# Génération données
with st.spinner("Génération des données..."):
    df = generate_lead_data(n_leads=n_leads)

# Section 1: Données
st.markdown("<h2>Dataset Overview</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Leads", f"{len(df):,}")
with col2:
    st.metric("Conversion Rate", f"{df['converted'].mean():.1%}")
with col3:
    st.metric("Average Time on Site", f"{df['time_on_site'].mean():.1f} min")

with st.expander("Data Preview"):
    st.dataframe(df.head(20), use_container_width=True)

# Section 2: Entraînement Modèle
st.markdown("<h2>Model Training & Evaluation</h2>", unsafe_allow_html=True)

model = LeadScorer(model_type=model_type)

with st.spinner("Training model..."):
    metrics, X_test, y_test, y_pred_proba = model.train(df)

# Métriques Performance
st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
with col2:
    st.metric("Precision", f"{metrics['precision']:.3f}")
with col3:
    st.metric("Recall", f"{metrics['recall']:.3f}")
with col4:
    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

st.info(f"Cross-validation AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")

# Visualisations métriques
col1, col2 = st.columns(2)

with col1:
    try:
        roc_fig = plot_roc_curve(y_test, y_pred_proba)
        st.plotly_chart(roc_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering ROC Curve: {str(e)}")

with col2:
    try:
        y_pred = (y_pred_proba > 0.5).astype(int)
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        st.plotly_chart(cm_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Confusion Matrix: {str(e)}")

# Feature importance
st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
try:
    fi_fig = plot_feature_importance(model.feature_importance)
    st.plotly_chart(fi_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering Feature Importance: {str(e)}")

# Section 3: Scoring Leads
st.markdown("<h2>Lead Scoring & Prioritization</h2>", unsafe_allow_html=True)

with st.spinner("Computing scores..."):
    scores_df = model.score_leads(df)

# Distribution scores
try:
    dist_fig = plot_score_distribution(scores_df)
    st.plotly_chart(dist_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering Score Distribution: {str(e)}")

# Statistiques par catégorie
st.markdown("<h3>Category Distribution</h3>", unsafe_allow_html=True)
category_stats = scores_df['categorie'].value_counts()

col1, col2, col3 = st.columns(3)
with col1:
    chaud = category_stats.get('Chaud', 0)
    st.metric("High Priority Leads", f"{chaud:,}", f"{chaud/len(scores_df):.1%}")
with col2:
    tiede = category_stats.get('Tiède', 0)
    st.metric("Medium Priority Leads", f"{tiede:,}", f"{tiede/len(scores_df):.1%}")
with col3:
    froid = category_stats.get('Froid', 0)
    st.metric("Low Priority Leads", f"{froid:,}", f"{froid/len(scores_df):.1%}")

# Top leads
st.markdown("<h3>Priority List - Top 20 Leads</h3>", unsafe_allow_html=True)
top_leads = scores_df.sort_values('score', ascending=False).head(20)

# Merge avec données originales
top_leads_full = top_leads.merge(df, on='lead_id')
st.dataframe(top_leads_full[['lead_id', 'score', 'categorie', 'time_on_site', 
                              'email_clicks', 'page_views', 'source']], use_container_width=True)

# Section 4: Impact Business
st.markdown("<h2>Business Impact & ROI</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    chauds = len(scores_df[scores_df['categorie'] == 'Chaud'])
    st.metric("High Priority Leads to Contact", f"{chauds:,}")

with col2:
    conversion_rate_chauds = 0.45
    conversions_estimated = int(chauds * conversion_rate_chauds)
    st.metric("Estimated Conversions (45%)", f"{conversions_estimated:,}")

with col3:
    avg_deal_value = 5000  # MAD
    revenue_potential = conversions_estimated * avg_deal_value
    st.metric("Revenue Potential", f"{revenue_potential:,} MAD")

st.caption("Assumptions: 45% conversion rate on high priority leads, 5,000 MAD average deal value")

# Footer
st.markdown("---")
st.markdown("**Projet réalisé par Bi Irié Albéric TRA** | École Centrale Casablanca | 2025")
st.markdown(f"Stack : Python, {model_type}, scikit-learn, Streamlit, Plotly")
