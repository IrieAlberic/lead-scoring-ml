"""
Fonctions métriques et visualisations lead scoring
"""
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, confusion_matrix
import pandas as pd
import numpy as np

def plot_roc_curve(y_test, y_pred_proba):
    """Courbe ROC"""
    # Convertir en arrays numpy
    y_test_array = np.array(y_test).flatten()
    y_pred_array = np.array(y_pred_proba).flatten()
    
    fpr, tpr, thresholds = roc_curve(y_test_array, y_pred_array)
    
    fig = go.Figure()
    
    # Ajouter la courbe ROC
    fig.add_trace(go.Scatter(
        x=list(fpr),
        y=list(tpr),
        mode='lines',
        name=f'ROC Curve (AUC={np.trapz(tpr, fpr):.2f})',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}'
    ))
    
    # Ajouter la ligne aléatoire
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash'),
        hovertemplate='x: %{x}<br>y: %{y}'
    ))
    
    fig.update_xaxes(title_text='False Positive Rate', range=[0, 1])
    fig.update_yaxes(title_text='True Positive Rate', range=[0, 1])
    fig.update_layout(
        title='Courbe ROC',
        hovermode='closest',
        template='plotly_white',
        height=450,
        width=600
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
    # Créer des bins manuellement
    bins = list(range(0, 105, 5))
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    
    fig = go.Figure()
    
    categories_list = ['Froid', 'Tiède', 'Chaud']
    colors = {'Chaud': '#28a745', 'Tiède': '#ffc107', 'Froid': '#6c757d'}
    
    for cat in categories_list:
        cat_data = scores_df[scores_df['categorie'] == cat]['score'].values
        if len(cat_data) > 0:
            # Créer l'histogramme
            counts, _ = np.histogram(cat_data, bins=bins)
            
            fig.add_trace(go.Bar(
                x=bin_centers,
                y=counts,
                name=cat,
                marker_color=colors.get(cat, '#cccccc'),
                hovertemplate='<b>%{fullData.name}</b><br>Score: %{x}<br>Nombre: %{y}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Distribution des Scores par Catégorie',
        xaxis_title='Score (0-100)',
        yaxis_title='Nombre de Leads',
        barmode='group',
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(range=[-2, 102])
    
    return fig

def plot_confusion_matrix(y_test, y_pred):
    """Matrice de confusion"""
    # Convertir en arrays numpy
    y_test_array = np.array(y_test).flatten()
    y_pred_array = np.array(y_pred).flatten()
    
    cm = confusion_matrix(y_test_array, y_pred_array)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm.tolist(),
        x=['Négatif', 'Positif'],
        y=['Réel Négatif', 'Réel Positif'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont=dict(size=16, color='white'),
        hovertemplate='%{y} - Prédit: %{x}<br>Count: %{z}'
    ))
    
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis_title='Valeur Prédite',
        yaxis_title='Valeur Réelle',
        template='plotly_white',
        height=450,
        width=600
    )
    
    return fig
