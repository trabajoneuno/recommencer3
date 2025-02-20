# modelo_recomendacion.py
import pandas as pd
from joblib import load

# 1. Primero defines la clase (DEBE ser exactamente igual a la original)
class ModeloRecomendacion:
    def __init__(self, similarity_df):
        self.similarity_df = similarity_df
    
    def recomendar_producto(self, producto):
        if producto not in self.similarity_df.columns:
            return {}
        similar_producto = self.similarity_df[producto].sort_values(ascending=False).head(6)
        similar_producto = similar_producto[1:]
        return dict(similar_producto)

# 2. Luego cargas el modelo
modelo = load('modelo_recomendacion.joblib')

# 3. Ya puedes usar el modelo
def obtener_recomendaciones(producto):
    try:
        recomendaciones = modelo.recomendar_producto(producto)
        return recomendaciones
    except Exception as e:
        print(f"Error al obtener recomendaciones: {str(e)}")
        return {}


