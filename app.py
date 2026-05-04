import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

app = FastAPI(title="Prédiction Arrêts Engins")

templates = Jinja2Templates(directory="templates")

# Charger les modèles et artéfacts au démarrage
model = None
encoders = None
features = None
history = None

@app.on_event("startup")
def load_artifacts():
    global model, encoders, features, history
    try:
        if os.path.exists('xgboost_model.pkl'):
            model = joblib.load('xgboost_model.pkl')
            encoders = joblib.load('encoders.pkl')
            features = joblib.load('features.pkl')
            if os.path.exists('history.pkl'):
                history = pd.read_pickle('history.pkl')
            print("Modèle chargé avec succès.")
        else:
            print("Aucun modèle trouvé. Veuillez exécuter data_pipeline.py d'abord.")
    except Exception as e:
        print(f"Erreur lors du chargement des modèles : {e}")

def get_quart(h):
    if 6 <= h < 14: return 'Matin'
    elif 14 <= h < 22: return 'Apres-midi'
    else: return 'Nuit'

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    engins = []
    types = []
    categories = []
    if encoders:
        if 'engin' in encoders: engins = encoders['engin'].classes_.tolist()
        if 'type' in encoders: types = encoders['type'].classes_.tolist()
        if 'categorie' in encoders: categories = encoders['categorie'].classes_.tolist()
        
    return templates.TemplateResponse(request=request, name="index.html", context={
        "engins": engins,
        "types": types,
        "categories": categories
    })

@app.post("/predict")
async def predict(
    request: Request,
    engin: str = Form(...),
    type: str = Form(...),
    categorie: str = Form(...),
    debut_arret: str = Form(...) # Format attendu : YYYY-MM-DDTHH:MM
):
    try:
        if not model:
            return templates.TemplateResponse(request=request, name="index.html", context={"error": "Modèle non chargé"})
            
        dt = pd.to_datetime(debut_arret)
        
        # Préparation de la donnée
        input_data = {
            'heure_jour': dt.hour,
            'jour_semaine': dt.dayofweek,
            'est_weekend': int(dt.dayofweek in [5, 6])
        }
        
        quart = get_quart(dt.hour)
        
        # Encodage
        try:
            if 'engin' in encoders: input_data['engin'] = encoders['engin'].transform([engin])[0]
            if 'type' in encoders: input_data['type'] = encoders['type'].transform([type])[0]
            if 'categorie' in encoders: input_data['categorie'] = encoders['categorie'].transform([categorie])[0]
            if 'quart_travail' in encoders: input_data['quart_travail'] = encoders['quart_travail'].transform([quart])[0]
        except ValueError as e:
            return templates.TemplateResponse("index.html", {"request": request, "error": f"Valeur inconnue : {e}"})
            
        # Features historiques
        tbf = 0.0
        moyenne = 0.0
        
        # Essayer de trouver l'engin avec son vrai nom pour chercher dans history
        # (Dans le pipeline, engin a été converti en string avant encodage, donc on cherche `engin` brut)
        if history is not None and engin in history.index:
            last_stop = history.loc[engin]
            tbf = (dt - last_stop['debut_arret']).total_seconds() / 3600.0
            tbf = max(0, tbf) # Si on prédit dans le passé par erreur
            moyenne = last_stop['duree_minutes']
            
        if 'temps_depuis_dernier_arret' in features:
            input_data['temps_depuis_dernier_arret'] = tbf
        if 'moyenne_duree_mobile_engin' in features:
            input_data['moyenne_duree_mobile_engin'] = moyenne
            
        # Construire le DataFrame final dans l'ordre des features
        df_input = pd.DataFrame([input_data])[features]
        
        # Prédiction
        pred = model.predict(df_input)[0]
        pred_minutes = max(0, float(pred))
        
        heures = int(pred_minutes // 60)
        minutes = int(pred_minutes % 60)
        
        result = f"{heures}h {minutes}m"
        
        # Re-passer les listes pour le formulaire
        engins = encoders['engin'].classes_.tolist() if 'engin' in encoders else []
        types = encoders['type'].classes_.tolist() if 'type' in encoders else []
        categories = encoders['categorie'].classes_.tolist() if 'categorie' in encoders else []
        
        return templates.TemplateResponse(request=request, name="index.html", context={
            "result": result,
            "engins": engins,
            "types": types,
            "categories": categories,
            "selected_engin": engin,
            "selected_type": type,
            "selected_categorie": categorie,
            "selected_date": debut_arret
        })
        
    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
