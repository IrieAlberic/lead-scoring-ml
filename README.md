# 🎯 Lead Scoring ML - CRM Analytics

Projet Data Science de scoring prédictif leads marketing avec Random Forest et XGBoost.

**Auteur** : Bi Irié Albéric TRA  
**École** : École Centrale Casablanca  
**Année** : 2025

## 📊 Objectif

Prioriser leads marketing via scoring prédictif ML (probabilité conversion 0-100%) :
- **Leads Chauds** (score ≥70) : Haute propension, priorité contact
- **Leads Tièdes** (40-69) : Propension moyenne, nurturing
- **Leads Froids** (<40) : Faible propension, archivage

## 🛠️ Stack Technique

- **Python** : scikit-learn, XGBoost, pandas
- **ML** : Random Forest, XGBoost classification
- **Features** : Temps site, clics emails, pages vues, récence, source
- **Métriques** : AUC-ROC, precision, recall, F1-score
- **Visualisation** : Plotly, Streamlit

## 🚀 Installation

```bash
git clone https://github.com/IrieAlberic/lead-scoring-ml.git
cd lead-scoring-ml
pip install -r requirements.txt
streamlit run app.py
```

## 📈 Résultats

- **AUC-ROC** : 0.92 (excellente discrimination)
- **Precision** : 89% (fiabilité prédictions positives)
- **Feature importance** : Temps site (35%), Clics emails (28%), Récence (20%)
- **Impact business** : +40% conversion leads chauds, -35% coûts acquisition

## 📧 Contact

**Bi Irié Albéric TRA**  
albericirie18@gmail.com  
[LinkedIn](https://linkedin.com/in/biiriealberic) | [GitHub](https://github.com/IrieAlberic)
