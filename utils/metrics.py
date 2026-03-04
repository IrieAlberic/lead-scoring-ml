"""
Fonctions métriques et visualisations lead scoring
"""
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, confusion_matrix
import pandas as pd

def plot_roc_curve(y_test, y_pred_proba):
    """Courbe ROC"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='#004F90', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Courbe ROC - Lead Scoring",
        xaxis_title="Taux Faux Positifs",
        yaxis_title="Taux Vrais Positifs",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_feature_importance(feature_importance_df):
    """Importance des features"""
    fig = px.bar(
        feature_importance_df.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Features - Importance",
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_score_distribution(scores_df):
    """Distribution scores leads"""
    fig = px.histogram(
        scores_df,
        x='score',
        color='categorie',
        title="Distribution Scores Leads par Catégorie",
        nbins=20,
        color_discrete_map={'Chaud': '#28a745', 'Tiède': '#ffc107', 'Froid': '#6c757d'}
    )
    
    fig.update_layout(
        xaxis_title="Score (0-100)",
        yaxis_title="Nombre de Leads",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_confusion_matrix(y_test, y_pred):
    """Matrice de confusion"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prédit Négatif', 'Prédit Positif'],
        y=['Réel Négatif', 'Réel Positif'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20}
    ))
    
    fig.update_layout(
        title="Matrice de Confusion",
        template="plotly_white",
        height=400
    )
    
    return fig
