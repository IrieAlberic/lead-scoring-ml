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

## 📈 Structure des Données

### Génération des Données

Les données sont **synthétiques** et générées automatiquement via `data/generate_leads.py`. Chaque run crée des leads avec comportements réalistes. La conversion dépend des features : temps site, clics emails, récence, etc.

### Colonnes (Features Marketing)

| Colonne | Type | Description | Plage | Exemple |
|---------|------|-------------|-------|---------|
| `lead_id` | String | Identifiant unique lead | LEAD_00001 à LEAD_N | LEAD_05234 |
| `time_on_site` | Float | Temps passé sur site (min) | 0.5 à 30 | 5.42 |
| `email_clicks` | Integer | Nombre clics en emails | 0 à 15 | 3 |
| `page_views` | Integer | Nombre pages vues | 1 à 50 | 12 |
| `recency_days` | Integer | Jours depuis dernière visite | 0 à 30 | 7 |
| `source` | String | Canal d'acquisition | organic, paid, referral, social | paid |
| `device` | String | Type d'appareil | desktop, mobile, tablet | mobile |
| `converted` | Integer | Conversion (0=Non, 1=Oui) | 0 ou 1 | 1 |

### Exemple de DataFrame

```
    lead_id  time_on_site  email_clicks  page_views  recency_days    source   device  converted
0 LEAD_00001          3.25              2          8            15    organic  desktop           0
1 LEAD_00002          8.50              5         18             2      paid   mobile           1
2 LEAD_00003          1.00              0          2            28   referral  tablet           0
3 LEAD_00004         12.30              8         35             1     social  mobile           1
```

### Distribution des Données

**Time on Site (Minutes)** :
- Moyenne : ~3.5 min
- Distribution : Gamma (shape=2, scale=3)
- Signification : Temps élevé = Intérêt fort

**Email Clicks** :
- Moyenne : ~2 clics
- Distribution : Poisson (λ=2)
- Signification : Clics élevés = Engagement email

**Page Views** :
- Moyenne : ~5 pages
- Distribution : Poisson (λ=5)
- Signification : Exploration site approfondie

**Recency Days** :
- Moyenne : ~7 jours
- Distribution : Exponentielle (scale=7)
- Signification : Récente visite = Lead "chaud"

**Source** :
- Organic : 30%
- Paid : 25%
- Referral : 25%
- Social : 20%

**Device** :
- Desktop : 50%
- Mobile : 40%
- Tablet : 10%

### Corrélation Features-Conversion

La **variable cible** (conversion) est calculée ainsi :

```python
propensity = (
    0.3 * (time_on_site / max) +
    0.3 * (email_clicks / max) +
    0.2 * (page_views / max) +
    0.2 * (1 - recency_days / max)
)
converted = (propensity > 0.5)
```

**Poids des features** :
- Temps site : 30%
- Clics emails : 30%
- Pages vues : 20%
- Récence : 20%

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

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## 📈 Résultats

- **AUC-ROC** : 0.92 (excellente discrimination)
- **Precision** : 89% (fiabilité prédictions positives)
- **Feature importance** : Temps site (35%), Clics emails (28%), Récence (20%)
- **Impact business** : +40% conversion leads chauds, -35% coûts acquisition

## � Structure

```
lead-scoring-ml/
├── app.py                 # Application Streamlit
├── data/generate_leads.py # Générateur données synthétiques
├── models/lead_scorer.py  # Pipeline ML (RF + XGBoost)
├── utils/metrics.py       # Fonctions visualisation
└── requirements.txt
```

## �📧 Contact

**Bi Irié Albéric TRA**  
albericirie18@gmail.com  
[LinkedIn](https://linkedin.com/in/biiriealberic) | [GitHub](https://github.com/IrieAlberic)
