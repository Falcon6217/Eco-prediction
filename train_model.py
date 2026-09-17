from pathlib import Path
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / 'ecosystem_data.csv'
MODEL = BASE / 'ecosystem.pkl'
FEATURES = ['water_quality','air_quality_index','biodiversity_index','vegetation_cover','soil_ph']


def train():
    df = pd.read_csv(DATA)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = FEATURES + ['ecosystem_health']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ecosystem_health'] = df['ecosystem_health'].astype(str).str.strip().str.lower()
    df = df.dropna(subset=required)
    df = df[df['ecosystem_health'].isin(['healthy','at risk','degraded'])]
    X = df[FEATURES]
    y = df['ecosystem_health'].map({'healthy':0,'at risk':1,'degraded':2})
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])
    model.fit(X, y)
    joblib.dump(model, MODEL)
    return len(df)

if __name__ == '__main__':
    print(f'Training rows: {train():,}')
    print(f'Model saved to: {MODEL.name}')
