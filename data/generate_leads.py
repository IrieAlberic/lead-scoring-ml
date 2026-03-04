"""
Générateur données synthétiques leads marketing
"""
import pandas as pd
import numpy as np

def generate_lead_data(n_leads=5000, seed=42):
    """
    Génère données synthétiques leads CRM
    
    Returns:
        DataFrame avec features comportementales + target converted
    """
    np.random.seed(seed)
    
    lead_ids = [f"LEAD_{i:05d}" for i in range(1, n_leads + 1)]
    
    # Features comportementales
    time_on_site = np.random.gamma(shape=2, scale=3, size=n_leads)  # minutes
    time_on_site = np.clip(time_on_site, 0.5, 30)
    
    email_clicks = np.random.poisson(lam=2, size=n_leads)
    email_clicks = np.clip(email_clicks, 0, 15)
    
    page_views = np.random.poisson(lam=5, size=n_leads)
    page_views = np.clip(page_views, 1, 50)
    
    recency_days = np.random.exponential(scale=7, size=n_leads).astype(int)
    recency_days = np.clip(recency_days, 0, 30)
    
    # Features catégorielles
    sources = np.random.choice(['organic', 'paid', 'referral', 'social'], size=n_leads, 
                              p=[0.3, 0.25, 0.25, 0.2])
    
    devices = np.random.choice(['desktop', 'mobile', 'tablet'], size=n_leads,
                              p=[0.5, 0.4, 0.1])
    
    # Target (conversion) - logique réaliste
    # Haute propension si : temps site élevé + clics emails élevés + récence faible
    propensity = (
        0.3 * (time_on_site / time_on_site.max()) +
        0.3 * (email_clicks / email_clicks.max()) +
        0.2 * (page_views / page_views.max()) +
        0.2 * (1 - recency_days / recency_days.max())
    )
    
    # Ajout bruit
    noise = np.random.normal(0, 0.15, size=n_leads)
    propensity = np.clip(propensity + noise, 0, 1)
    
    # Conversion binaire (seuil 0.5)
    converted = (propensity > 0.5).astype(int)
    
    df = pd.DataFrame({
        'lead_id': lead_ids,
        'time_on_site': time_on_site.round(2),
        'email_clicks': email_clicks,
        'page_views': page_views,
        'recency_days': recency_days,
        'source': sources,
        'device': devices,
        'converted': converted
    })
    
    return df

if __name__ == "__main__":
    df = generate_lead_data()
    print(df.head())
    print(f"\nConversion rate: {df['converted'].mean():.2%}")
