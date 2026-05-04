import pandas as pd

try:
    df = pd.read_excel('Arrêts engins modifié_2025.xlsx', skiprows=8)
    print("Columns:")
    print(df.columns.tolist())
    print("\nFirst row:")
    print(df.head(1).to_dict(orient='records'))
except Exception as e:
    print(f"Error: {e}")
