# main.py
import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
import os

# Definición del modelo
class ModeloRecomendacion:
    def __init__(self, similarity_df):
        self.similarity_df = similarity_df
    
    def recomendar_producto(self, producto):
        if producto not in self.similarity_df.columns:
            return {}
        similar_producto = self.similarity_df[producto].sort_values(ascending=False).head(6)
        similar_producto = similar_producto[1:]
        return dict(similar_producto)

# Obtener la ruta absoluta al archivo joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, 'modelo_recomendacion.joblib')

# Cargar el modelo
try:
    modelo = load(modelo_path)
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    modelo = None

# Función de recomendación
def obtener_recomendaciones(producto):
    if modelo is None:
        return {"error": "Modelo no disponible"}
    try:
        recomendaciones = modelo.recomendar_producto(producto)
        return recomendaciones
    except Exception as e:
        print(f"Error al obtener recomendaciones: {str(e)}")
        return {}

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Recomendaciones",
    description="API para recomendar productos similares",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "API de Recomendaciones activa"}

@app.get("/recomendar/{producto}")
async def recomendar(producto: str):
    recomendaciones = obtener_recomendaciones(producto)
    if not recomendaciones:
        raise HTTPException(
            status_code=404,
            detail="Producto no encontrado o error en las recomendaciones"
        )
    return recomendaciones

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
