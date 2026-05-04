import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_and_clean_data(file_path):
    print("Loading data...")
    # Read the data, skip first 8 rows
    df = pd.read_excel(file_path, skiprows=8)
    
    # Standardize column names
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
    
    print("Initial shape:", df.shape)
    
    # 2.1 Traitement de la variable cible (Durée)
    if 'durée' in df.columns:
        df.rename(columns={'durée': 'duree'}, inplace=True)
    if 'début_arrêt' in df.columns:
        df.rename(columns={'début_arrêt': 'debut_arret'}, inplace=True)
    if 'fin_arrêt' in df.columns:
        df.rename(columns={'fin_arrêt': 'fin_arret'}, inplace=True)
    if 'catégorie' in df.columns:
        df.rename(columns={'catégorie': 'categorie'}, inplace=True)
        
    return df

def process_features(df):
    print("Processing features...")
    # Convert dates
    df['debut_arret'] = pd.to_datetime(df['debut_arret'], errors='coerce')
    df['fin_arret'] = pd.to_datetime(df['fin_arret'], errors='coerce')
    
    # Drop rows without dates
    df = df.dropna(subset=['debut_arret'])
    
    # Calculate duree_minutes if not directly parseable
    def parse_duration(d):
        if pd.isna(d):
            return 0
        if isinstance(d, str):
            parts = d.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60.0
        elif isinstance(d, pd.Timedelta):
            return d.total_seconds() / 60.0
        return 0
        
    if 'duree' in df.columns:
        df['duree_minutes'] = df['duree'].apply(parse_duration)
    else:
        # Fallback: calculate from dates
        df['duree_minutes'] = (df['fin_arret'] - df['debut_arret']).dt.total_seconds() / 60.0
    
    # Remove negative durations (anomalies)
    df = df[df['duree_minutes'] >= 0]
    
    # 2.2 Missing values
    if 'type' in df.columns:
        df['type'] = df['type'].fillna('Inconnu')
    if 'categorie' in df.columns:
        df['categorie'] = df['categorie'].fillna('Inconnu')
    
    # 3.1 Time features
    df['heure_jour'] = df['debut_arret'].dt.hour
    df['jour_semaine'] = df['debut_arret'].dt.dayofweek
    df['est_weekend'] = df['jour_semaine'].isin([5, 6]).astype(int)
    
    def get_quart(h):
        if 6 <= h < 14: return 'Matin'
        elif 14 <= h < 22: return 'Apres-midi'
        else: return 'Nuit'
    df['quart_travail'] = df['heure_jour'].apply(get_quart)
    
    # 3.2 Historical features
    if 'engin' in df.columns:
        df = df.sort_values(by=['engin', 'debut_arret'])
        
        # TBF (Time Between Failures)
        df['temps_depuis_dernier_arret'] = df.groupby('engin')['debut_arret'].diff().dt.total_seconds() / 3600.0
        df['temps_depuis_dernier_arret'] = df['temps_depuis_dernier_arret'].fillna(0) # First stop
        
        # Moving average of duration
        df['moyenne_duree_mobile_engin'] = df.groupby('engin')['duree_minutes'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
        df['moyenne_duree_mobile_engin'] = df['moyenne_duree_mobile_engin'].fillna(df['duree_minutes'].mean())
    
    # Sort chronologically for TimeSeriesSplit
    df = df.sort_values('debut_arret').reset_index(drop=True)
    
    return df

def encode_and_train(df):
    print("Encoding and training...")
    # Encodage des variables catégorielles
    cat_cols = ['engin', 'type', 'categorie', 'quart_travail']
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Convert to string to avoid mixed type errors
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
    # Features & Target
    features = ['heure_jour', 'jour_semaine', 'est_weekend']
    for col in cat_cols:
        if col in df.columns: features.append(col)
    if 'temps_depuis_dernier_arret' in df.columns: features.append('temps_depuis_dernier_arret')
    if 'moyenne_duree_mobile_engin' in df.columns: features.append('moyenne_duree_mobile_engin')
    
    X = df[features]
    y = df['duree_minutes']
    
    # Time Series Split (80% train, 20% test roughly)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Baseline: Random Forest
    print("Training Baseline (Random Forest)...")
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(f"RF MAE: {mean_absolute_error(y_test, rf_pred):.2f}")
    print(f"RF RMSE: {root_mean_squared_error(y_test, rf_pred):.2f}")
    print(f"RF R² (Accuracy): {r2_score(y_test, rf_pred):.4f}")
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42,
        objective='reg:squarederror'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    print(f"XGB MAE: {mean_absolute_error(y_test, xgb_pred):.2f}")
    print(f"XGB RMSE: {root_mean_squared_error(y_test, xgb_pred):.2f}")
    print(f"XGB R² (Accuracy): {r2_score(y_test, xgb_pred):.4f}")
    
    # Save models and encoders
    print("Saving models...")
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(features, 'features.pkl')
    
    # Compute historical aggregates to use in API for moving averages / TBF
    print("Saving historical aggregates...")
    if 'engin' in df.columns:
        last_stops = df.groupby('engin').last()[['debut_arret', 'duree_minutes']]
        last_stops.to_pickle('history.pkl')
        
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    df_raw = load_and_clean_data('Arrêts engins modifié_2025.xlsx')
    df_processed = process_features(df_raw)
    encode_and_train(df_processed)
