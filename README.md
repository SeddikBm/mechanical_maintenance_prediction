# Prédiction des Arrêts d'Engins Miniers

Ce projet propose un modèle de Machine Learning (XGBoost) pour prédire la durée d'inactivité des engins miniers dès l'instant où un arrêt est signalé.

## Architecture

1. **Pipeline de données (`data_pipeline.py`)** : Charge l'historique des arrêts depuis le fichier Excel, effectue le nettoyage des dates, crée des features temporelles et historiques (temps moyen, TBF), et entraîne un modèle XGBoost.
2. **Modèles générés** : `xgboost_model.pkl`, `encoders.pkl`, `features.pkl`, `history.pkl`.
3. **API et Frontend (`app.py`)** : Une API FastAPI servant une interface HTML basique pour faire des prédictions.
4. **Conteneurisation (`Dockerfile`)** : Prêt pour le déploiement.

## Installation

### Mode Local
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python data_pipeline.py
uvicorn app:app --host 0.0.0.0 --port 8000
```
Puis ouvrez `http://localhost:8000` dans votre navigateur.

### Mode Docker
```bash
docker build -t prediction-arrets .
docker run -p 8000:8000 prediction-arrets
```
Puis ouvrez `http://localhost:8000` dans votre navigateur.

## Performances du Modèle

Sur l'ensemble de test :
- Random Forest MAE : 183.89 minutes
- XGBoost MAE : 200.42 minutes
