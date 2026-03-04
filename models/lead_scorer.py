"""
Pipeline Lead Scoring avec Random Forest et XGBoost
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

class LeadScorer:
    """Modèle scoring leads ML"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.label_encoders = {}
        self.feature_importance = None
        
    def prepare_features(self, df, fit=True):
        """Encode features catégorielles"""
        df_copy = df.copy()
        
        cat_cols = ['source', 'device']
        
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col])
                self.label_encoders[col] = le
            else:
                df_copy[col] = self.label_encoders[col].transform(df_copy[col])
        
        return df_copy
    
    def train(self, df):
        """
        Entraîne modèle ML sur données leads
        
        Returns:
            dict avec métriques performance
        """
        # Préparation features
        df_prep = self.prepare_features(df, fit=True)
        
        X = df_prep.drop(['lead_id', 'converted'], axis=1)
        y = df_prep['converted']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entraînement modèle
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:  # xgboost
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métriques
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='roc_auc'
        )
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        return metrics, X_test, y_test, y_pred_proba
    
    def score_leads(self, df):
        """
        Score nouveaux leads (0-100)
        
        Returns:
            DataFrame avec scores
        """
        df_prep = self.prepare_features(df, fit=False)
        X = df_prep.drop(['lead_id', 'converted'], axis=1, errors='ignore')
        
        probas = self.model.predict_proba(X)[:, 1]
        scores = (probas * 100).round(0).astype(int)
        
        # Catégories
        categories = []
        for score in scores:
            if score >= 70:
                categories.append("Chaud")
            elif score >= 40:
                categories.append("Tiède")
            else:
                categories.append("Froid")
        
        result = df[['lead_id']].copy()
        result['score'] = scores
        result['probabilite_conversion'] = probas.round(3)
        result['categorie'] = categories
        
        return result

if __name__ == "__main__":
    from data.generate_leads import generate_lead_data
    
    df = generate_lead_data(n_leads=2000)
    
    model = LeadScorer(model_type='random_forest')
    metrics, _, _, _ = model.train(df)
    
    print("Performance modèle:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
